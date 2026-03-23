"""
prepare.py — VisDrone Task 1 (Object Detection in Images) data preparation
for YOLOv12 / autoresearch-style framework.

Role (mirrors karpathy/autoresearch prepare.py):
  - One-time dataset setup: convert VisDrone annotations → YOLO-format TXT labels
  - Anchor analysis: k-means on GT boxes + IoU coverage statistics
  - Runtime utilities: dataset class, dataloader factory, evaluation helpers

VisDrone directory structure expected (after manual download from aiskyeye.com):
    <VISDRONE_ROOT>/
        VisDrone2019-DET-train/
            images/          *.jpg
            annotations/     *.txt   (VisDrone format)
        VisDrone2019-DET-val/
            images/
            annotations/
        VisDrone2019-DET-test-dev/
            images/
            annotations/
        VisDrone2019-DET-testset-challenge/
            images/
            annotations/

Run once to prepare:
    python prepare.py --root /path/to/visdrone [--anchors 9] [--imgsz 640]

After preparation, YOLO-format labels land alongside images:
    VisDrone2019-DET-train/labels/*.txt
    VisDrone2019-DET-val/labels/*.txt
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants (fixed — do not modify in train.py)
# ---------------------------------------------------------------------------

# VisDrone Task 1 categories
# Index 0 = ignored region, 11 = others → skipped in YOLO labels
VISDRONE_CLASSES = {
    1:  "pedestrian",
    2:  "people",
    3:  "bicycle",
    4:  "car",
    5:  "van",
    6:  "truck",
    7:  "tricycle",
    8:  "awning-tricycle",
    9:  "bus",
    10: "motor",
}
NUM_CLASSES = len(VISDRONE_CLASSES)        # 10
CLASS_NAMES = [VISDRONE_CLASSES[i] for i in sorted(VISDRONE_CLASSES)]

# Remap VisDrone category id (1-10) → YOLO class index (0-9)
VISDRONE_TO_YOLO = {vid: (vid - 1) for vid in VISDRONE_CLASSES}

# Default model input size (can be overridden via CLI / train.py)
IMG_SIZE = 640

# Anchor analysis defaults
DEFAULT_NUM_ANCHORS = 9    # 3 anchors × 3 feature-map scales (YOLOv5-style ref)
ANCHOR_IOU_THRESHOLD = 0.25  # "good" anchor coverage threshold (YOLO auto-anchor)

# Cache / output
CACHE_DIR = Path(os.path.expanduser("~")) / ".cache" / "visdrone_yolo"


# ---------------------------------------------------------------------------
# VisDrone annotation parser
# ---------------------------------------------------------------------------

def parse_visdrone_annotation(ann_path: Path):
    """
    Parse a single VisDrone annotation file.

    VisDrone format (one object per line):
        bbox_left, bbox_top, bbox_width, bbox_height,
        score, object_category, truncation, occlusion

    score in GT: 1 = consider in eval, 0 = ignore
    object_category: 0 = ignored region, 11 = others (both skipped)

    Returns
    -------
    list of dict with keys:
        x1, y1, w, h  — pixel coords (absolute)
        category      — VisDrone category id (1-10)
        truncation    — 0/1/-1
        occlusion     — 0/1/2/-1
    """
    objects = []
    if not ann_path.exists():
        return objects
    with open(ann_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 8:
                continue
            try:
                x1, y1, w, h = (int(parts[i]) for i in range(4))
                score = int(parts[4])
                cat   = int(parts[5])
                trunc = int(parts[6])
                occl  = int(parts[7])
            except ValueError:
                continue

            # Skip ignored regions (cat=0), "others" (cat=11), and score=0 entries
            if cat not in VISDRONE_CLASSES:
                continue
            if score == 0:
                continue
            if w <= 0 or h <= 0:
                continue

            objects.append(dict(x1=x1, y1=y1, w=w, h=h,
                                category=cat, truncation=trunc, occlusion=occl))
    return objects


def visdrone_to_yolo_label(objects, img_w: int, img_h: int):
    """
    Convert parsed VisDrone objects to YOLO label lines.

    YOLO format (normalised, centre-based):
        class_idx  cx  cy  bw  bh
    all values in [0, 1].
    """
    lines = []
    for obj in objects:
        cls = VISDRONE_TO_YOLO[obj["category"]]
        cx = (obj["x1"] + obj["w"] / 2.0) / img_w
        cy = (obj["y1"] + obj["h"] / 2.0) / img_h
        bw = obj["w"] / img_w
        bh = obj["h"] / img_h
        # Clamp to valid range
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        bw = max(0.0, min(1.0, bw))
        bh = max(0.0, min(1.0, bh))
        if bw > 0 and bh > 0:
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


# ---------------------------------------------------------------------------
# Dataset conversion (one-time)
# ---------------------------------------------------------------------------

def convert_split(split_dir: Path, verbose: bool = True):
    """
    Convert VisDrone annotation TXTs to YOLO label TXTs for one split.

    split_dir/
        images/    <image files>
        annotations/  <annotation TXTs>
      → creates:
        split_dir/labels/  <YOLO label TXTs>
    """
    img_dir  = split_dir / "images"
    ann_dir  = split_dir / "annotations"
    lbl_dir  = split_dir / "labels"
    lbl_dir.mkdir(exist_ok=True)

    img_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not img_files:
        print(f"  [warn] No images found in {img_dir}", file=sys.stderr)
        return 0

    n_converted = 0
    n_skipped   = 0
    for img_path in img_files:
        ann_path = ann_dir / (img_path.stem + ".txt")
        lbl_path = lbl_dir / (img_path.stem + ".txt")

        if lbl_path.exists():
            n_converted += 1
            continue  # already done

        try:
            with Image.open(img_path) as im:
                img_w, img_h = im.size
        except Exception as e:
            print(f"  [warn] Cannot open image {img_path}: {e}", file=sys.stderr)
            n_skipped += 1
            continue

        objects = parse_visdrone_annotation(ann_path)
        lines   = visdrone_to_yolo_label(objects, img_w, img_h)

        with open(lbl_path, "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

        n_converted += 1

    if verbose:
        print(f"  {split_dir.name}: {n_converted} images converted, "
              f"{n_skipped} skipped → labels in {lbl_dir}")
    return n_converted


# ---------------------------------------------------------------------------
# Anchor ↔ GT IoU analysis
# ---------------------------------------------------------------------------

def collect_gt_wh(split_dir: Path, img_size: int = IMG_SIZE):
    """
    Collect all (width, height) of ground-truth boxes from a split,
    normalised to [0,1] and then scaled to img_size pixels.

    This mirrors the YOLOv5 auto-anchor procedure: bboxes are
    recalculated relative to the letterboxed model input size.

    Returns
    -------
    np.ndarray  shape (N, 2)  — width, height in *pixels at img_size*
    """
    lbl_dir = split_dir / "labels"
    img_dir = split_dir / "images"

    wh_list = []
    for lbl_path in sorted(lbl_dir.glob("*.txt")):
        img_path = img_dir / (lbl_path.stem + ".jpg")
        if not img_path.exists():
            img_path = img_dir / (lbl_path.stem + ".png")
        if not img_path.exists():
            continue

        try:
            with Image.open(img_path) as im:
                iw, ih = im.size
        except Exception:
            continue

        # Letterbox scale factor (keep aspect ratio, pad to square)
        scale = img_size / max(iw, ih)

        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                # bw, bh are normalised [0,1] → multiply by original size → scale
                bw = float(parts[3]) * iw * scale
                bh = float(parts[4]) * ih * scale
                if bw > 0 and bh > 0:
                    wh_list.append([bw, bh])

    return np.array(wh_list, dtype=np.float32) if wh_list else np.zeros((0, 2), dtype=np.float32)


def iou_wh(anchors: np.ndarray, gt_wh: np.ndarray) -> np.ndarray:
    """
    Compute IoU between every (anchor_w, anchor_h) pair and every GT box,
    assuming both are centred at the origin (width/height IoU only).

    Parameters
    ----------
    anchors : (A, 2)  — anchor widths & heights
    gt_wh   : (N, 2)  — GT box widths & heights

    Returns
    -------
    iou     : (N, A)  — IoU matrix
    """
    # Intersection: min(w_a, w_gt) * min(h_a, h_gt)
    # A  →  (1, A, 2)   GT  →  (N, 1, 2)
    a = anchors[np.newaxis, :, :]    # (1, A, 2)
    g = gt_wh[:, np.newaxis, :]      # (N, 1, 2)

    inter_w = np.minimum(a[..., 0], g[..., 0])
    inter_h = np.minimum(a[..., 1], g[..., 1])
    inter   = inter_w * inter_h                  # (N, A)

    area_a  = a[..., 0] * a[..., 1]             # (1, A)
    area_g  = g[..., 0] * g[..., 1]             # (N, 1)
    union   = area_a + area_g - inter

    return inter / (union + 1e-9)               # (N, A)


def kmeans_anchors(gt_wh: np.ndarray, n_anchors: int = DEFAULT_NUM_ANCHORS,
                   n_iter: int = 300, seed: int = 42) -> np.ndarray:
    """
    K-means clustering on GT box sizes using 1-IoU as distance metric,
    matching the original YOLOv2 / YOLOv5 anchor generation procedure.

    Parameters
    ----------
    gt_wh     : (N, 2)
    n_anchors : number of anchor clusters
    n_iter    : maximum iterations

    Returns
    -------
    anchors   : (n_anchors, 2)  sorted by area ascending
    """
    assert len(gt_wh) >= n_anchors, \
        f"Need at least {n_anchors} GT boxes, got {len(gt_wh)}"

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(gt_wh), n_anchors, replace=False)
    centroids = gt_wh[idx].copy()

    for _ in range(n_iter):
        dist  = 1.0 - iou_wh(centroids, gt_wh)          # (N, A)
        assignments = dist.argmin(axis=1)                # (N,)

        new_centroids = np.zeros_like(centroids)
        changed = False
        for k in range(n_anchors):
            mask = assignments == k
            if mask.sum() == 0:
                # Re-init dead cluster to random GT box
                new_centroids[k] = gt_wh[rng.integers(len(gt_wh))]
                changed = True
            else:
                new_centroids[k] = gt_wh[mask].mean(axis=0)
                if not np.allclose(new_centroids[k], centroids[k], atol=0.01):
                    changed = True

        centroids = new_centroids
        if not changed:
            break

    # Sort by area (w*h)
    areas = centroids[:, 0] * centroids[:, 1]
    centroids = centroids[np.argsort(areas)]
    return centroids


def anchor_coverage(anchors: np.ndarray, gt_wh: np.ndarray,
                    thr: float = ANCHOR_IOU_THRESHOLD):
    """
    Compute anchor coverage statistics:
      - Best-anchor IoU per GT box (max over anchors)
      - Fraction of GT boxes whose best-anchor IoU ≥ thr  ("coverage ratio")
      - Mean best-anchor IoU

    Parameters
    ----------
    anchors : (A, 2)
    gt_wh   : (N, 2)
    thr     : IoU threshold for "good" match

    Returns
    -------
    dict with keys: best_iou (N,), coverage_ratio, mean_best_iou
    """
    if len(gt_wh) == 0:
        return dict(best_iou=np.array([]), coverage_ratio=0.0, mean_best_iou=0.0)

    iou = iou_wh(anchors, gt_wh)          # (N, A)
    best_iou = iou.max(axis=1)            # (N,)
    coverage = (best_iou >= thr).mean()
    return dict(
        best_iou=best_iou,
        coverage_ratio=float(coverage),
        mean_best_iou=float(best_iou.mean()),
    )


def analyse_anchors(split_dir: Path, n_anchors: int = DEFAULT_NUM_ANCHORS,
                    img_size: int = IMG_SIZE, verbose: bool = True):
    """
    Full anchor analysis pipeline for a dataset split:
      1. Collect GT box sizes (scaled to img_size)
      2. Run k-means to find n_anchors cluster centres
      3. Compute IoU coverage statistics

    Returns
    -------
    dict with:
        anchors        : (n_anchors, 2) float32 — suggested anchor sizes (px)
        coverage_ratio : float  — fraction of GT boxes matched (IoU ≥ thr)
        mean_best_iou  : float
        gt_wh          : (N, 2) — raw GT widths/heights
    """
    if verbose:
        print(f"\n[Anchor analysis] {split_dir.name}, img_size={img_size}, "
              f"n_anchors={n_anchors}")

    gt_wh = collect_gt_wh(split_dir, img_size=img_size)
    if len(gt_wh) == 0:
        print("  [warn] No GT boxes found — run convert_split first?",
              file=sys.stderr)
        return {}

    if verbose:
        print(f"  GT boxes collected: {len(gt_wh):,}")
        print(f"  WH stats  min=({gt_wh[:,0].min():.1f},{gt_wh[:,1].min():.1f})  "
              f"median=({np.median(gt_wh[:,0]):.1f},{np.median(gt_wh[:,1]):.1f})  "
              f"max=({gt_wh[:,0].max():.1f},{gt_wh[:,1].max():.1f})")

    anchors = kmeans_anchors(gt_wh, n_anchors=n_anchors)
    stats   = anchor_coverage(anchors, gt_wh)

    if verbose:
        print(f"\n  Suggested {n_anchors} anchors (w×h px at {img_size}px input):")
        for i, (w, h) in enumerate(anchors):
            print(f"    [{i:2d}]  {w:6.1f} × {h:6.1f}")
        print(f"\n  Coverage (best-anchor IoU ≥ {ANCHOR_IOU_THRESHOLD}):")
        print(f"    ratio       : {stats['coverage_ratio']*100:.1f}%")
        print(f"    mean best   : {stats['mean_best_iou']:.4f}")

        # Per-decile breakdown
        deciles = np.percentile(stats["best_iou"], [10, 25, 50, 75, 90])
        labels  = ["p10", "p25", "p50", "p75", "p90"]
        print("  Best-IoU percentiles: " +
              "  ".join(f"{l}={v:.3f}" for l, v in zip(labels, deciles)))

    return dict(
        anchors=anchors,
        coverage_ratio=stats["coverage_ratio"],
        mean_best_iou=stats["mean_best_iou"],
        gt_wh=gt_wh,
    )


# ---------------------------------------------------------------------------
# Runtime Dataset & DataLoader  (used by train.py)
# ---------------------------------------------------------------------------

class VisDroneDetDataset(Dataset):
    """
    Lightweight VisDrone detection dataset for YOLOv12 training.

    Loads images and YOLO-format labels.  Applies basic letterbox resize.
    Does NOT apply mosaic / augmentation (leave that to train.py / albumentations).

    Parameters
    ----------
    split_dir : Path  — e.g. .../VisDrone2019-DET-train
    img_size  : int   — square input size (default 640)
    max_labels: int   — pad/truncate label tensor to this length
    augment   : bool  — if True, random hflip (extendable in train.py)
    """

    def __init__(self, split_dir: Path | str, img_size: int = IMG_SIZE,
                 max_labels: int = 500, augment: bool = False):
        self.split_dir  = Path(split_dir)
        self.img_size   = img_size
        self.max_labels = max_labels
        self.augment    = augment

        img_dir = self.split_dir / "images"
        lbl_dir = self.split_dir / "labels"

        self.samples: list[tuple[Path, Path]] = []
        for ext in ("*.jpg", "*.png"):
            for ip in sorted(img_dir.glob(ext)):
                lp = lbl_dir / (ip.stem + ".txt")
                self.samples.append((ip, lp))

        if not self.samples:
            raise FileNotFoundError(
                f"No images found in {img_dir}.  "
                "Did you run: python prepare.py --root <VISDRONE_ROOT> ?"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]

        # ---- Load image ----
        img = Image.open(img_path).convert("RGB")
        iw, ih = img.size

        # Letterbox
        scale = self.img_size / max(iw, ih)
        nw, nh = int(round(iw * scale)), int(round(ih * scale))
        img_resized = img.resize((nw, nh), Image.BILINEAR)

        canvas = Image.new("RGB", (self.img_size, self.img_size), (114, 114, 114))
        pad_x = (self.img_size - nw) // 2
        pad_y = (self.img_size - nh) // 2
        canvas.paste(img_resized, (pad_x, pad_y))

        # To tensor  [C, H, W]  float32  in [0, 1]
        img_t = torch.from_numpy(
            np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0
        )

        # ---- Load labels ----
        labels = []   # each: [cls, cx, cy, bw, bh]  in normalised canvas coords
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls = int(parts[0])
                    cx, cy, bw, bh = (float(p) for p in parts[1:])
                    # Adjust for letterbox padding
                    cx = (cx * nw + pad_x) / self.img_size
                    cy = (cy * nh + pad_y) / self.img_size
                    bw = bw * nw / self.img_size
                    bh = bh * nh / self.img_size
                    labels.append([cls, cx, cy, bw, bh])

        # ---- Random hflip augmentation ----
        if self.augment and random.random() < 0.5:
            img_t = img_t.flip(-1)
            labels = [[c, 1.0 - cx, cy, bw, bh] for c, cx, cy, bw, bh in labels]

        # ---- Pad / truncate label tensor ----
        lbl_t = torch.zeros((self.max_labels, 5), dtype=torch.float32)
        if labels:
            arr = torch.tensor(labels[:self.max_labels], dtype=torch.float32)
            lbl_t[:len(arr)] = arr

        return img_t, lbl_t, torch.tensor(min(len(labels), self.max_labels))


def collate_fn(batch):
    imgs, labels, counts = zip(*batch)
    return torch.stack(imgs), torch.stack(labels), torch.stack(counts)


def get_dataloader(split_dir: Path | str, img_size: int = IMG_SIZE,
                   batch_size: int = 16, num_workers: int = 4,
                   augment: bool = False, shuffle: bool = True):
    """
    Factory for train / val DataLoaders.

    Usage in train.py:
        from prepare import get_dataloader
        train_loader = get_dataloader(TRAIN_DIR, batch_size=16, augment=True)
        val_loader   = get_dataloader(VAL_DIR,   batch_size=16, shuffle=False)
    """
    ds = VisDroneDetDataset(split_dir, img_size=img_size, augment=augment)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=shuffle,
    )


# ---------------------------------------------------------------------------
# Evaluation helpers  (used by train.py)
#
# Two independent metrics:
#
#   1. val_box_iou  — mean IoU of each predicted box against its best-matched
#                     GT box (across all images).  Measures localisation quality
#                     independent of class.  Range [0, 1], higher is better.
#
#   2. val_cls_acc  — classification accuracy of matched predictions:
#                     among pred-GT pairs whose IoU ≥ MATCH_IOU_THRESHOLD,
#                     what fraction have the correct class label?
#                     Range [0, 1], higher is better.
#
# Both functions share the same pred / GT tensor convention:
#   pred_boxes_list : list[Tensor]  each (P, 6)  — cx, cy, w, h, conf, cls
#   gt_boxes_list   : list[Tensor]  each (G, 5)  — cls, cx, cy, w, h
#   (normalised coords in [0, 1]; class indices 0-based)
#
# The combined evaluate() function runs both and returns a dict.
# ---------------------------------------------------------------------------

# IoU threshold for calling a prediction "matched" to a GT box
MATCH_IOU_THRESHOLD = 0.50


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """(cx, cy, w, h) → (x1, y1, x2, y2)"""
    x1 = boxes[..., 0] - boxes[..., 2] / 2
    y1 = boxes[..., 1] - boxes[..., 3] / 2
    x2 = boxes[..., 0] + boxes[..., 2] / 2
    y2 = boxes[..., 1] + boxes[..., 3] / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_iou_batch(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Pairwise IoU between two sets of boxes in (x1, y1, x2, y2) format.

    boxes1 : (N, 4)
    boxes2 : (M, 4)
    Returns: (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * \
            (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * \
            (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter = (inter_x2 - inter_x1).clamp(min=0) * \
            (inter_y2 - inter_y1).clamp(min=0)           # (N, M)
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-9)


# ---------------------------------------------------------------------------
# Metric 1 — Box IoU
# ---------------------------------------------------------------------------

def compute_box_iou(
    pred_boxes_list: list[torch.Tensor],
    gt_boxes_list:   list[torch.Tensor],
) -> float:
    """
    Mean IoU of every prediction against its best-matched GT box.

    For each predicted box we find the GT box with the highest IoU
    (regardless of class).  We accumulate all those best-IoU values and
    return their mean over the whole validation set.

    A perfect detector scores 1.0; a detector that finds no overlapping
    boxes at all scores 0.0.  Unmatched GT boxes (false negatives) are
    NOT penalised here — this metric focuses purely on localisation of
    the detections that were made.

    Parameters
    ----------
    pred_boxes_list : one Tensor per image  shape (P, 6)  cx cy w h conf cls
    gt_boxes_list   : one Tensor per image  shape (G, 5)  cls cx cy w h

    Returns
    -------
    val_box_iou : float  in [0, 1]
    """
    iou_scores: list[float] = []

    for preds, gts in zip(pred_boxes_list, gt_boxes_list):
        if preds is None or len(preds) == 0:
            continue
        if gts is None or len(gts) == 0:
            # All predictions are false positives — IoU = 0 for each
            iou_scores.extend([0.0] * len(preds))
            continue

        p_xyxy = cxcywh_to_xyxy(preds[:, :4])          # (P, 4)
        g_xyxy = cxcywh_to_xyxy(gts[:, 1:5])           # (G, 4)
        iou    = box_iou_batch(p_xyxy, g_xyxy)          # (P, G)

        best_iou_per_pred = iou.max(dim=1).values       # (P,)
        iou_scores.extend(best_iou_per_pred.tolist())

    return float(np.mean(iou_scores)) if iou_scores else 0.0


# ---------------------------------------------------------------------------
# Metric 2 — Classification Accuracy
# ---------------------------------------------------------------------------

def compute_cls_accuracy(
    pred_boxes_list: list[torch.Tensor],
    gt_boxes_list:   list[torch.Tensor],
    match_iou_thr:   float = MATCH_IOU_THRESHOLD,
) -> dict:
    """
    Classification accuracy among IoU-matched prediction–GT pairs.

    Matching procedure (greedy, confidence-sorted):
      1. Sort predictions by confidence (highest first).
      2. For each prediction, find the unmatched GT box with the highest IoU.
      3. If that IoU ≥ match_iou_thr, form a match and mark the GT as used.
      4. A matched prediction is "correct" if pred_cls == gt_cls.

    Returns
    -------
    dict with keys:
        val_cls_acc        : float — overall accuracy of matched pairs
        per_class_acc      : dict[class_name → float]
        n_matched          : int   — total matched pairs
        n_gt               : int   — total GT boxes
        n_pred             : int   — total predictions
    """
    n_correct  = 0
    n_matched  = 0
    n_gt_total = 0
    n_pd_total = 0

    # Per-class counters
    cls_correct = np.zeros(NUM_CLASSES, dtype=np.int64)
    cls_total   = np.zeros(NUM_CLASSES, dtype=np.int64)

    for preds, gts in zip(pred_boxes_list, gt_boxes_list):
        has_pred = preds is not None and len(preds) > 0
        has_gt   = gts   is not None and len(gts)   > 0

        n_gt_total += (len(gts)   if has_gt   else 0)
        n_pd_total += (len(preds) if has_pred else 0)

        if not has_pred or not has_gt:
            continue

        # Sort predictions by confidence descending
        order = preds[:, 4].argsort(descending=True)
        preds_sorted = preds[order]

        p_xyxy   = cxcywh_to_xyxy(preds_sorted[:, :4])  # (P, 4)
        g_xyxy   = cxcywh_to_xyxy(gts[:, 1:5])          # (G, 4)
        iou_mat  = box_iou_batch(p_xyxy, g_xyxy)         # (P, G)

        matched_gt = torch.zeros(len(gts), dtype=torch.bool)

        for pi in range(len(preds_sorted)):
            # Mask out already-matched GT boxes
            iou_row = iou_mat[pi].clone()
            iou_row[matched_gt] = -1.0

            best_iou, best_gi = iou_row.max(0)
            if best_iou < match_iou_thr:
                continue  # no valid GT match for this prediction

            matched_gt[best_gi] = True
            n_matched += 1

            pred_cls = int(preds_sorted[pi, 5].item())
            gt_cls   = int(gts[best_gi, 0].item())

            cls_total[gt_cls] += 1
            if pred_cls == gt_cls:
                n_correct += 1
                cls_correct[gt_cls] += 1

    overall_acc = n_correct / n_matched if n_matched > 0 else 0.0

    per_class_acc = {}
    for c in range(NUM_CLASSES):
        name = CLASS_NAMES[c]
        per_class_acc[name] = (
            float(cls_correct[c] / cls_total[c]) if cls_total[c] > 0 else float("nan")
        )

    return dict(
        val_cls_acc   = float(overall_acc),
        per_class_acc = per_class_acc,
        n_matched     = n_matched,
        n_gt          = n_gt_total,
        n_pred        = n_pd_total,
    )


# ---------------------------------------------------------------------------
# Combined evaluation entry-point  (call this from train.py)
# ---------------------------------------------------------------------------

def evaluate(
    pred_boxes_list: list[torch.Tensor],
    gt_boxes_list:   list[torch.Tensor],
    match_iou_thr:   float = MATCH_IOU_THRESHOLD,
    verbose:         bool  = False,
) -> dict:
    """
    Run both evaluation metrics and return a unified result dict.

    Typical usage in train.py
    -------------------------
        from prepare import evaluate

        all_preds, all_gts = [], []
        model.eval()
        with torch.no_grad():
            for imgs, labels, counts in val_loader:
                preds = model(imgs.to(device))   # (B, P, 6): cx cy w h conf cls
                for b in range(len(imgs)):
                    n = counts[b].item()
                    all_preds.append(preds[b])
                    all_gts.append(labels[b, :n])

        metrics = evaluate(all_preds, all_gts)
        print(f"val_box_iou: {metrics['val_box_iou']:.4f}")
        print(f"val_cls_acc: {metrics['val_cls_acc']:.4f}")

    Parameters
    ----------
    pred_boxes_list : list[Tensor (P, 6)]  — cx cy w h conf cls
    gt_boxes_list   : list[Tensor (G, 5)]  — cls cx cy w h
    match_iou_thr   : IoU threshold for matching preds to GTs (default 0.50)
    verbose         : if True, print per-class breakdown

    Returns
    -------
    dict with keys:
        val_box_iou   : float  — mean best-GT IoU per prediction  [0, 1]
        val_cls_acc   : float  — classification accuracy of matched pairs  [0, 1]
        per_class_acc : dict[class_name → float]
        n_matched     : int
        n_gt          : int
        n_pred        : int
    """
    box_iou  = compute_box_iou(pred_boxes_list, gt_boxes_list)
    cls_info = compute_cls_accuracy(pred_boxes_list, gt_boxes_list,
                                    match_iou_thr=match_iou_thr)

    result = dict(val_box_iou=box_iou, **cls_info)

    if verbose:
        print(f"\n[Evaluation]  IoU threshold for matching = {match_iou_thr}")
        print(f"  Metric 1 — val_box_iou : {box_iou:.4f}  "
              f"(mean best-GT IoU over {cls_info['n_pred']} predictions)")
        print(f"  Metric 2 — val_cls_acc : {cls_info['val_cls_acc']:.4f}  "
              f"({cls_info['n_matched']} matched / {cls_info['n_gt']} GT boxes)")
        print(f"\n  Per-class classification accuracy:")
        for name, acc in cls_info["per_class_acc"].items():
            bar = ("█" * int(acc * 20)).ljust(20) if not np.isnan(acc) else " " * 20
            acc_str = f"{acc:.3f}" if not np.isnan(acc) else "  N/A "
            print(f"    {name:>18s}  {acc_str}  |{bar}|")

    return result


# ---------------------------------------------------------------------------
# CLI: one-time dataset preparation + anchor analysis
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare VisDrone Task-1 dataset for YOLOv12"
    )
    parser.add_argument(
        "--root", type=str, required=True,
        help="Root directory containing VisDrone2019-DET-{train,val} sub-folders"
    )
    parser.add_argument(
        "--imgsz", type=int, default=IMG_SIZE,
        help=f"Model input image size (default: {IMG_SIZE})"
    )
    parser.add_argument(
        "--anchors", type=int, default=DEFAULT_NUM_ANCHORS,
        help=f"Number of anchor clusters for k-means (default: {DEFAULT_NUM_ANCHORS})"
    )
    parser.add_argument(
        "--anchor-analysis", action="store_true",
        help="Run k-means anchor analysis on train split (optional)"
    )
    args = parser.parse_args()

    root = Path(args.root)
    splits = {
        "train":               root / "VisDrone2019-DET-train",
        "val":                 root / "VisDrone2019-DET-val",
        "test-dev":            root / "VisDrone2019-DET-test-dev",
        "testset-challenge":   root / "VisDrone2019-DET-testset-challenge",
    }

    # ---- Step 1: convert annotations ----
    print("=" * 60)
    print("Step 1: Converting VisDrone annotations → YOLO labels")
    print("=" * 60)
    for name, split_dir in splits.items():
        if not split_dir.exists():
            print(f"  [skip] {split_dir} not found")
            continue
        convert_split(split_dir, verbose=True)

    # ---- Step 2: anchor analysis on train split ----
    if args.anchor_analysis:
        train_dir = splits["train"]
        if train_dir.exists() and (train_dir / "labels").exists():
            print("\n" + "=" * 60)
            print("Step 2: Anchor analysis (k-means on GT boxes)")
            print("=" * 60)
            result = analyse_anchors(
                train_dir,
                n_anchors=args.anchors,
                img_size=args.imgsz,
            )
            if result:
                anchors_px = result["anchors"]
                print("\n  Anchor sizes normalised to 1.0 (for YAML config):")
                for w, h in anchors_px:
                    print(f"    [{w/args.imgsz:.4f}, {h/args.imgsz:.4f}]")

                # Save anchor suggestions to file
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                anchor_path = CACHE_DIR / "suggested_anchors.txt"
                with open(anchor_path, "w") as f:
                    f.write(f"# Anchors for VisDrone Task-1, img_size={args.imgsz}\n")
                    f.write(f"# coverage_ratio={result['coverage_ratio']:.4f}  "
                            f"mean_best_iou={result['mean_best_iou']:.4f}\n")
                    for w, h in anchors_px:
                        f.write(f"{w:.2f},{h:.2f}\n")
                print(f"\n  Anchors saved to {anchor_path}")
        else:
            print("\n[skip] Anchor analysis: train labels not ready.")

    # ---- Step 3: dataset statistics report ----
    print("\n" + "=" * 60)
    print("Step 3: Dataset statistics")
    print("=" * 60)
    for split_name, split_dir in splits.items():
        lbl_dir = split_dir / "labels"
        img_dir = split_dir / "images"
        if not lbl_dir.exists():
            print(f"  [skip] {split_name}: labels not found")
            continue

        img_count = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png")))
        cls_count = np.zeros(NUM_CLASSES, dtype=np.int64)
        box_count = 0

        for lp in lbl_dir.glob("*.txt"):
            with open(lp) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_count[int(parts[0])] += 1
                        box_count += 1

        print(f"\n  [{split_name}]  images={img_count:,}  total_boxes={box_count:,}")
        print(f"  {'Class':<20s}  {'Count':>7s}  {'Ratio':>6s}  Distribution")
        print(f"  {'-'*20}  {'-'*7}  {'-'*6}  {'-'*24}")
        for c, cname in enumerate(CLASS_NAMES):
            ratio = cls_count[c] / box_count if box_count > 0 else 0.0
            bar   = "█" * int(ratio * 48)
            print(f"  {cname:<20s}  {cls_count[c]:>7,}  {ratio:>5.1%}  {bar}")

    print("\nSetup complete.  You can now run: python train_simple.py")


if __name__ == "__main__":
    main()
