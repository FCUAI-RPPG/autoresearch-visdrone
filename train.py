"""
train.py — YOLOv12 training on VisDrone Task-1 (Object Detection in Images)
Agent modifies this file to experiment with architecture / hyperparameters.

Fixed contract with prepare.py:
  - get_dataloader(split_dir, img_size, batch_size, augment, shuffle)
      returns DataLoader yielding (imgs, labels, counts)
      imgs   : (B, 3, H, W)  float32  [0,1]
      labels : (B, MAX_LABELS, 5)  float32  [cls, cx, cy, bw, bh]
      counts : (B,)  int  — number of valid GT boxes per image

  - evaluate(pred_boxes_list, gt_boxes_list, verbose=False)
      returns dict with val_box_iou (float) and val_cls_acc (float)

  - NUM_CLASSES, IMG_SIZE, CLASS_NAMES  (constants)

Metrics printed at the end of each run (grep-friendly):
    val_box_iou: 0.XXXX
    val_cls_acc: 0.XXXX
    peak_vram_mb: XXXX
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    CLASS_NAMES,
    IMG_SIZE,
    NUM_CLASSES,
    evaluate,
    get_dataloader,
)

# ===========================================================================
# ① Hyperparameters — agent tweaks these
# ===========================================================================

VISDRONE_ROOT   = Path(os.environ.get("VISDRONE_ROOT", "/data/visdrone"))
TRAIN_DIR       = VISDRONE_ROOT / "VisDrone2019-DET-train"
VAL_DIR         = VISDRONE_ROOT / "VisDrone2019-DET-val"
TEST_DIR        = VISDRONE_ROOT / "VisDrone2019-DET-test-dev"
CHALLENGE_DIR   = VISDRONE_ROOT / "VisDrone2019-DET-testset-challenge"

IMG_SIZE        = 640           # model input resolution
BATCH_SIZE      = 8             # images per step
NUM_WORKERS     = 4

LR              = 1e-3          # initial learning rate (AdamW)
WEIGHT_DECAY    = 1e-4
WARMUP_STEPS    = 500           # linear LR warm-up
MAX_GRAD_NORM   = 10.0

# Detection head
CONF_THRESHOLD  = 0.25          # objectness threshold at eval time
NMS_IOU_THR     = 0.45          # NMS IoU threshold at eval time

# Loss weights
LAMBDA_BOX      = 7.5           # CIoU box loss weight   (same as v8)
LAMBDA_DFL      = 1.5           # DFL loss weight        (same as v8)
LAMBDA_CLS      = 0.5           # Classification BCE weight (same as v8)

# Fixed time budget (seconds) — mirrors autoresearch
TIME_BUDGET     = 300           # 5 minutes, wall-clock training time


# ===========================================================================
# ② Model — YOLOv12-style anchor-free detector
#    Agent may modify depth / width multipliers or swap blocks.
# ===========================================================================

# Depth / width multipliers (nano-scale defaults)
DEPTH_MULTIPLE  = 0.33
WIDTH_MULTIPLE  = 0.25
BASE_CHANNELS   = 64            # scaled by WIDTH_MULTIPLE


def _make_divisible(v: float, div: int = 8) -> int:
    return max(div, int(v + div / 2) // div * div)


def ch(c: int) -> int:
    """Apply width multiplier and round to nearest 8."""
    return _make_divisible(c * WIDTH_MULTIPLE)


def rep(n: int) -> int:
    """Apply depth multiplier and round up."""
    return max(1, round(n * DEPTH_MULTIPLE))


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class Conv(nn.Module):
    """Conv + BN + SiLU."""
    def __init__(self, c_in, c_out, k=1, s=1, p=None):
        super().__init__()
        p = p if p is not None else k // 2
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(c_out, eps=1e-3, momentum=0.03)
        self.act  = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard residual bottleneck."""
    def __init__(self, c, shortcut=True):
        super().__init__()
        self.cv1 = Conv(c, c, 3)
        self.cv2 = Conv(c, c, 3)
        self.add = shortcut

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """
    CSP-style block used in YOLOv8/v12.
    Splits channels, runs N bottleneck stacks, then concatenates.
    """
    def __init__(self, c_in, c_out, n=1, shortcut=True):
        super().__init__()
        c_mid = c_out // 2
        self.cv1 = Conv(c_in, c_out, 1)
        self.cv2 = Conv((2 + n) * c_mid, c_out, 1)
        self.m   = nn.ModuleList(
            Bottleneck(c_mid, shortcut) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast."""
    def __init__(self, c_in, c_out, k=5):
        super().__init__()
        c_  = c_in // 2
        self.cv1 = Conv(c_in, c_, 1)
        self.cv2 = Conv(c_ * 4, c_out, 1)
        self.pool = nn.MaxPool2d(k, stride=1, padding=k // 2)

    def forward(self, x):
        x  = self.cv1(x)
        p1 = self.pool(x)
        p2 = self.pool(p1)
        p3 = self.pool(p2)
        return self.cv2(torch.cat([x, p1, p2, p3], dim=1))


# ---------------------------------------------------------------------------
# Backbone  (CSPDarkNet-like, 3 output strides: P3/8, P4/16, P5/32)
# ---------------------------------------------------------------------------

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        # Stem
        self.stem  = Conv(3, ch(BASE_CHANNELS), 3, 2)          # /2

        # Stage 2  → /4
        self.s2    = nn.Sequential(
            Conv(ch(BASE_CHANNELS),      ch(BASE_CHANNELS * 2), 3, 2),
            C2f(ch(BASE_CHANNELS * 2),   ch(BASE_CHANNELS * 2), rep(3)),
        )
        # Stage 3  → /8   (P3)
        self.s3    = nn.Sequential(
            Conv(ch(BASE_CHANNELS * 2),  ch(BASE_CHANNELS * 4), 3, 2),
            C2f(ch(BASE_CHANNELS * 4),   ch(BASE_CHANNELS * 4), rep(6)),
        )
        # Stage 4  → /16  (P4)
        self.s4    = nn.Sequential(
            Conv(ch(BASE_CHANNELS * 4),  ch(BASE_CHANNELS * 8), 3, 2),
            C2f(ch(BASE_CHANNELS * 8),   ch(BASE_CHANNELS * 8), rep(6)),
        )
        # Stage 5  → /32  (P5)
        self.s5    = nn.Sequential(
            Conv(ch(BASE_CHANNELS * 8),  ch(BASE_CHANNELS * 16), 3, 2),
            C2f(ch(BASE_CHANNELS * 16),  ch(BASE_CHANNELS * 16), rep(3)),
            SPPF(ch(BASE_CHANNELS * 16), ch(BASE_CHANNELS * 16)),
        )

    def forward(self, x):
        x  = self.stem(x)
        x  = self.s2(x)
        p3 = self.s3(x)
        p4 = self.s4(p3)
        p5 = self.s5(p4)
        return p3, p4, p5


# ---------------------------------------------------------------------------
# Neck  (BiFPN-lite / PANet up-down path)
# ---------------------------------------------------------------------------

class Neck(nn.Module):
    def __init__(self):
        super().__init__()
        c3 = ch(BASE_CHANNELS * 4)
        c4 = ch(BASE_CHANNELS * 8)
        c5 = ch(BASE_CHANNELS * 16)

        # Top-down path
        self.lat5  = Conv(c5, c4, 1)
        self.td4   = C2f(c4 * 2, c4, rep(3), shortcut=False)

        self.lat4  = Conv(c4, c3, 1)
        self.td3   = C2f(c3 * 2, c3, rep(3), shortcut=False)

        # Bottom-up path
        self.dn3   = Conv(c3, c3, 3, 2)
        self.bu4   = C2f(c3 + c4, c4, rep(3), shortcut=False)

        self.dn4   = Conv(c4, c4, 3, 2)
        self.bu5   = C2f(c4 + c5, c5, rep(3), shortcut=False)

        self.out_channels = (c3, c4, c5)

    def forward(self, feats):
        p3, p4, p5 = feats

        # Top-down
        p4_td = self.td4(torch.cat([
            F.interpolate(self.lat5(p5), scale_factor=2, mode="nearest"), p4
        ], dim=1))
        p3_td = self.td3(torch.cat([
            F.interpolate(self.lat4(p4_td), scale_factor=2, mode="nearest"), p3
        ], dim=1))

        # Bottom-up
        p4_bu = self.bu4(torch.cat([self.dn3(p3_td), p4_td], dim=1))
        p5_bu = self.bu5(torch.cat([self.dn4(p4_bu), p5],    dim=1))

        return p3_td, p4_bu, p5_bu


# ---------------------------------------------------------------------------
# Detection Head  (anchor-free, decoupled cls / reg)
# ---------------------------------------------------------------------------

REG_MAX = 16        # DFL bins


class DFL(nn.Module):
    """Distribution Focal Loss integral projection."""
    def __init__(self, reg_max: int = REG_MAX):
        super().__init__()
        self.reg_max = reg_max
        self.conv = nn.Conv2d(reg_max, 1, 1, bias=False)
        x = torch.arange(reg_max, dtype=torch.float32)
        self.conv.weight.data[:] = x.view(1, reg_max, 1, 1)
        self.conv.weight.requires_grad_(False)

    def forward(self, x):
        # x : (B, 4*reg_max, H, W)
        B, _, H, W = x.shape
        x = x.view(B, 4, self.reg_max, H * W)
        x = x.softmax(2)
        x = self.conv(x.permute(0, 1, 3, 2).reshape(B * 4, self.reg_max, H * W, 1))
        return x.view(B, 4, H * W)


class DetectHead(nn.Module):
    """
    Anchor-free decoupled detection head.
    Outputs per-scale: (B, 4*REG_MAX + NUM_CLASSES, H, W)
    """
    def __init__(self, in_channels: tuple[int, ...]):
        super().__init__()
        self.nl = len(in_channels)   # number of scales (3)
        self.reg_max = REG_MAX
        c_reg = max(16, in_channels[0] // 4, 4 * REG_MAX)
        c_cls = max(in_channels[0], NUM_CLASSES)

        self.reg_convs = nn.ModuleList(
            nn.Sequential(Conv(c, c_reg, 3), Conv(c_reg, c_reg, 3),
                          nn.Conv2d(c_reg, 4 * REG_MAX, 1))
            for c in in_channels
        )
        self.cls_convs = nn.ModuleList(
            nn.Sequential(Conv(c, c_cls, 3), Conv(c_cls, c_cls, 3),
                          nn.Conv2d(c_cls, NUM_CLASSES, 1))
            for c in in_channels
        )
        self.dfl = DFL(REG_MAX)

    def forward(self, feats: list[torch.Tensor]):
        raw = []
        for i, x in enumerate(feats):
            reg = self.reg_convs[i](x)   # (B, 4*REG_MAX, H, W)
            cls = self.cls_convs[i](x)   # (B, NUM_CLASSES, H, W)
            raw.append(torch.cat([reg, cls], dim=1))
        return raw   # list of (B, 4*REG_MAX+NC, Hi, Wi)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class YOLOv12(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.neck     = Neck()
        self.head     = DetectHead(self.neck.out_channels)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Bias init for cls head: prior prob = 0.01
        prior = math.log(0.01 / (1 - 0.01))
        for conv_seq in self.head.cls_convs:
            nn.init.constant_(conv_seq[-1].bias, prior)

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.neck(feats)
        return self.head(feats)   # list of 3 raw tensors


# ===========================================================================
# ③ Loss — YOLOv12 / v8 official pipeline
#
# Exactly mirrors ultralytics v8DetectionLoss:
#
#   Assigner : TaskAlignedAssigner
#               align_metric = cls_score^alpha * iou^beta   (α=0.5, β=6.0)
#               topk=10 candidates per GT (inside-GT-box only)
#               one-to-one conflict resolution via max-overlap
#
#   Box loss  : CIoU  weighted by normalised align_metric per positive
#   DFL loss  : Distribution Focal Loss on ltrb bin distribution
#   Cls loss  : BCE(logits, iou_score * one_hot)  — Varifocal-style soft target
#
#   Total     : box_gain*L_box + dfl_gain*L_dfl + cls_gain*L_cls
#               (no separate objectness term — same as v8/v12)
# ===========================================================================

# ---------------------------------------------------------------------------
# Utility: anchor point generation & box decoding
# ---------------------------------------------------------------------------

def make_anchors(feats: list[torch.Tensor], img_size: int, device,
                 grid_cell_offset: float = 0.5):
    """
    Generate (anchor_points, stride_tensor) from feature-map sizes.
    anchor_points : (A, 2)  in pixel coords at img_size
    stride_tensor : (A, 1)
    """
    pts, strides = [], []
    for feat in feats:
        H, W = feat.shape[2], feat.shape[3]
        sh = img_size / H          # stride for this scale
        sy = torch.arange(H, device=device, dtype=torch.float32) + grid_cell_offset
        sx = torch.arange(W, device=device, dtype=torch.float32) + grid_cell_offset
        gy, gx = torch.meshgrid(sy, sx, indexing="ij")
        # pixel coords
        pts.append(torch.stack([gx.flatten() * (img_size / W),
                                 gy.flatten() * (img_size / H)], dim=1))
        strides.append(torch.full((H * W, 1), sh, device=device))
    return torch.cat(pts), torch.cat(strides)          # (A,2), (A,1)


def decode_preds(raw_list: list[torch.Tensor], img_size: int,
                 anchor_points: torch.Tensor, stride_tensor: torch.Tensor):
    """
    Decode raw head outputs into (pred_bboxes_xyxy, pred_scores, pred_dist_raw).

    pred_bboxes_xyxy : (B, A, 4)  pixel xyxy at img_size
    pred_scores      : (B, A, NC) raw logits
    pred_dist_raw    : (B, A, 4*REG_MAX) raw DFL logits (for DFL loss)
    """
    device = raw_list[0].device
    B = raw_list[0].shape[0]

    dist_list, cls_list = [], []
    for raw in raw_list:
        H, W = raw.shape[2], raw.shape[3]
        n = H * W
        flat = raw.permute(0, 2, 3, 1).reshape(B, n, -1)
        dist_list.append(flat[..., :4 * REG_MAX])
        cls_list.append(flat[..., 4 * REG_MAX:])

    pred_dist_raw = torch.cat(dist_list, dim=1)   # (B, A, 4*REG_MAX)
    pred_scores   = torch.cat(cls_list,  dim=1)   # (B, A, NC)

    # DFL → ltrb distances (in pixels)
    # (B, A, 4, REG_MAX) → softmax → weighted sum
    dist = pred_dist_raw.view(B, -1, 4, REG_MAX)
    bins = torch.arange(REG_MAX, device=device, dtype=torch.float32)
    ltrb = dist.softmax(-1).matmul(bins)           # (B, A, 4)

    # ltrb → xyxy   anchor_points in pixel coords
    ap = anchor_points[None]                        # (1, A, 2)
    x1 = ap[..., 0] - ltrb[..., 0]
    y1 = ap[..., 1] - ltrb[..., 1]
    x2 = ap[..., 0] + ltrb[..., 2]
    y2 = ap[..., 1] + ltrb[..., 3]
    pred_bboxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)   # (B, A, 4)

    return pred_bboxes_xyxy, pred_scores, pred_dist_raw


# ---------------------------------------------------------------------------
# IoU helpers (pixel xyxy)
# ---------------------------------------------------------------------------

def bbox_iou_xyxy(b1: torch.Tensor, b2: torch.Tensor, eps: float = 1e-7):
    """
    Pairwise IoU between two sets of xyxy boxes.
    b1 : (N, 4)   b2 : (M, 4)   →   (N, M)
    """
    area1 = (b1[:, 2] - b1[:, 0]).clamp(0) * (b1[:, 3] - b1[:, 1]).clamp(0)
    area2 = (b2[:, 2] - b2[:, 0]).clamp(0) * (b2[:, 3] - b2[:, 1]).clamp(0)
    ix1 = torch.max(b1[:, None, 0], b2[None, :, 0])
    iy1 = torch.max(b1[:, None, 1], b2[None, :, 1])
    ix2 = torch.min(b1[:, None, 2], b2[None, :, 2])
    iy2 = torch.min(b1[:, None, 3], b2[None, :, 3])
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    return inter / (area1[:, None] + area2[None, :] - inter + eps)


def ciou_elementwise(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-7):
    """
    Element-wise CIoU loss. Both tensors: (N, 4) pixel xyxy.
    Returns (N,) loss values.
    """
    pw, ph = (pred[:, 2] - pred[:, 0]).clamp(0), (pred[:, 3] - pred[:, 1]).clamp(0)
    gw, gh = (gt[:, 2]   - gt[:, 0]).clamp(0),   (gt[:, 3]   - gt[:, 1]).clamp(0)
    pcx = (pred[:, 0] + pred[:, 2]) / 2
    pcy = (pred[:, 1] + pred[:, 3]) / 2
    gcx = (gt[:, 0]   + gt[:, 2])   / 2
    gcy = (gt[:, 1]   + gt[:, 3])   / 2

    inter_w = (torch.min(pred[:, 2], gt[:, 2]) - torch.max(pred[:, 0], gt[:, 0])).clamp(0)
    inter_h = (torch.min(pred[:, 3], gt[:, 3]) - torch.max(pred[:, 1], gt[:, 1])).clamp(0)
    inter   = inter_w * inter_h
    union   = pw * ph + gw * gh - inter + eps
    iou     = inter / union

    # Enclosing diagonal²
    cw = torch.max(pred[:, 2], gt[:, 2]) - torch.min(pred[:, 0], gt[:, 0])
    ch = torch.max(pred[:, 3], gt[:, 3]) - torch.min(pred[:, 1], gt[:, 1])
    c2 = cw ** 2 + ch ** 2 + eps

    rho2 = (pcx - gcx) ** 2 + (pcy - gcy) ** 2

    v = (4 / math.pi ** 2) * (torch.atan(gw / (gh + eps)) -
                               torch.atan(pw / (ph + eps))) ** 2
    with torch.no_grad():
        alpha_ciou = v / (1 - iou + v + eps)

    return 1.0 - iou + rho2 / c2 + alpha_ciou * v


# ---------------------------------------------------------------------------
# DFL loss (Distribution Focal Loss)
# ---------------------------------------------------------------------------

def dfl_loss(pred_dist: torch.Tensor, target_ltrb: torch.Tensor) -> torch.Tensor:
    """
    DFL loss.
    pred_dist   : (N, 4, REG_MAX)  raw logits before softmax
    target_ltrb : (N, 4)           continuous target distances in [0, REG_MAX-1]
    Returns mean scalar.
    """
    tgt = target_ltrb.clamp(0, REG_MAX - 1 - 0.01)   # (N, 4)
    tl  = tgt.long()                                  # floor bin
    tr  = tl + 1                                      # ceil  bin
    wl  = tr.float() - tgt                            # weight for floor
    wr  = 1.0 - wl                                    # weight for ceil

    # Cross-entropy on each side, weighted by the complementary fraction
    # pred_dist : (N, 4, REG_MAX)
    loss = (
        F.cross_entropy(pred_dist.view(-1, REG_MAX),
                        tl.clamp(0, REG_MAX - 1).view(-1), reduction="none")
        * wl.view(-1)
        +
        F.cross_entropy(pred_dist.view(-1, REG_MAX),
                        tr.clamp(0, REG_MAX - 1).view(-1), reduction="none")
        * wr.view(-1)
    )
    return loss.view(-1, 4).mean(-1).mean()     # scalar


# ---------------------------------------------------------------------------
# TaskAlignedAssigner  (self-contained, no ultralytics dependency)
#
# Exactly matches ultralytics TaskAlignedAssigner with
#   topk=10, alpha=0.5, beta=6.0
# ---------------------------------------------------------------------------

class TaskAlignedAssigner(nn.Module):
    """
    Assign GT boxes to anchor points using the task-aligned metric.

    align_metric = cls_score^alpha * iou^beta

    For each GT box:
      1. Keep only anchors whose centre lies inside the GT box.
      2. Among those, take the top-k by align_metric.
      3. If an anchor is assigned to multiple GTs, keep the one
         with the highest overlap.

    Parameters
    ----------
    topk       : candidates per GT  (default 10, same as v8/v12)
    num_classes: number of classes
    alpha      : classification exponent  (default 0.5)
    beta       : IoU exponent             (default 6.0)
    """

    def __init__(self, topk: int = 10, num_classes: int = NUM_CLASSES,
                 alpha: float = 0.5, beta: float = 6.0, eps: float = 1e-9):
        super().__init__()
        self.topk = topk
        self.nc   = num_classes
        self.alpha = alpha
        self.beta  = beta
        self.eps   = eps

    @torch.no_grad()
    def forward(self,
                pd_scores:  torch.Tensor,   # (B, A, NC)  sigmoid probabilities
                pd_bboxes:  torch.Tensor,   # (B, A, 4)   xyxy pixels
                anc_points: torch.Tensor,   # (A, 2)      cx cy pixels
                gt_labels:  torch.Tensor,   # (B, G)      int class ids
                gt_bboxes:  torch.Tensor,   # (B, G, 4)   xyxy pixels
                mask_gt:    torch.Tensor,   # (B, G)      bool valid
                ):
        """
        Returns
        -------
        target_labels  : (B, A)      assigned GT class  (nc = background)
        target_bboxes  : (B, A, 4)   assigned GT xyxy box
        target_scores  : (B, A, NC)  soft classification target
        fg_mask        : (B, A)      bool foreground
        target_gt_idx  : (B, A)      index into GT for each positive anchor
        """
        B, G = gt_labels.shape
        device = pd_scores.device

        # ── Step 1: compute align metric for every (anchor, GT) pair ──
        # iou_matrix : (B, G, A)
        iou_mat = torch.zeros(B, G, pd_bboxes.shape[1], device=device)
        align   = torch.zeros(B, G, pd_bboxes.shape[1], device=device)

        for b in range(B):
            valid = mask_gt[b]            # (G,)  bool
            if not valid.any():
                continue
            g_boxes = gt_bboxes[b][valid]   # (g, 4)
            p_boxes = pd_bboxes[b]          # (A, 4)
            g_cls   = gt_labels[b][valid]   # (g,)

            iou = bbox_iou_xyxy(g_boxes, p_boxes)   # (g, A)

            # cls score for each GT class : (g, A)
            cls_s = pd_scores[b, :, g_cls].T         # (g, A)

            am = cls_s.pow(self.alpha) * iou.pow(self.beta)

            # place into full-G tensors
            iou_mat[b, valid] = iou
            align[b, valid]   = am

        # ── Step 2: mask anchors that lie inside each GT box ──
        # inside_gt : (B, G, A)
        ax = anc_points[None, None, :, 0]    # (1,1,A)
        ay = anc_points[None, None, :, 1]
        inside_gt = (
            (ax > gt_bboxes[:, :, None, 0]) &
            (ay > gt_bboxes[:, :, None, 1]) &
            (ax < gt_bboxes[:, :, None, 2]) &
            (ay < gt_bboxes[:, :, None, 3])
        )                                    # (B, G, A)
        inside_gt = inside_gt & mask_gt[:, :, None]

        align = align * inside_gt.float()   # zero outside GT boxes

        # ── Step 3: top-k per GT ──
        topk_k = min(self.topk, align.shape[2])
        topk_vals, _ = align.topk(topk_k, dim=2, largest=True)
        topk_thr = topk_vals[:, :, -1:].clamp(min=self.eps)
        mask_topk = (align >= topk_thr) & inside_gt   # (B, G, A)

        # ── Step 4: resolve anchor assigned to multiple GTs ──
        # If an anchor matches >1 GT, keep the GT with the highest iou
        count = mask_topk.sum(dim=1)            # (B, A) — how many GTs want it
        multi = count > 1                       # (B, A) conflict anchors

        if multi.any():
            max_gt_idx = iou_mat.argmax(dim=1)  # (B, A) best GT by IoU
            # one-hot : (B, G, A)
            one_hot = F.one_hot(max_gt_idx, G).permute(0, 2, 1).float()
            # for conflicted anchors, replace mask_topk with one_hot
            conflict_mask = multi[:, None, :].expand_as(mask_topk)
            mask_topk = torch.where(conflict_mask, one_hot.bool(), mask_topk)

        # ── Step 5: build targets ──
        fg_mask     = mask_topk.any(dim=1)          # (B, A)
        target_gt_idx = mask_topk.float().argmax(dim=1)   # (B, A)

        # target_labels
        target_labels = torch.full((B, pd_scores.shape[1]), self.nc,
                                    dtype=torch.long, device=device)
        for b in range(B):
            fg = fg_mask[b]
            if fg.any():
                target_labels[b, fg] = gt_labels[b][target_gt_idx[b, fg]]

        # target_bboxes
        target_bboxes = torch.zeros(B, pd_scores.shape[1], 4, device=device)
        for b in range(B):
            fg = fg_mask[b]
            if fg.any():
                target_bboxes[b, fg] = gt_bboxes[b][target_gt_idx[b, fg]]

        # target_scores: soft label = iou * one_hot  (Varifocal-style)
        target_scores = torch.zeros(B, pd_scores.shape[1], self.nc, device=device)
        for b in range(B):
            fg = fg_mask[b]
            if fg.any():
                gt_c  = target_labels[b, fg]                      # (P,)
                iou_s = iou_mat[b, target_gt_idx[b, fg],
                                torch.where(fg)[0]]               # (P,)
                iou_s = iou_s.clamp(0, 1)
                target_scores[b, fg] = F.one_hot(gt_c, self.nc).float() * \
                                       iou_s[:, None]

        return target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx


# ---------------------------------------------------------------------------
# Main loss function  (v8DetectionLoss equivalent)
# ---------------------------------------------------------------------------

# Singleton assigner (created once, reused every step)
_assigner = None

def get_assigner(device) -> TaskAlignedAssigner:
    global _assigner
    if _assigner is None or next(iter(_assigner.parameters() if hasattr(_assigner, 'parameters') else []), None) is None:
        _assigner = TaskAlignedAssigner(
            topk=10, num_classes=NUM_CLASSES, alpha=0.5, beta=6.0
        ).to(device)
    return _assigner


def compute_loss(raw_list: list[torch.Tensor],
                 labels:   torch.Tensor,
                 counts:   torch.Tensor,
                 img_size: int,
                 device) -> tuple[torch.Tensor, dict]:
    """
    YOLOv12 / v8 detection loss.

    Parameters
    ----------
    raw_list : list of 3 head tensors (B, 4*REG_MAX+NC, H, W)
    labels   : (B, MAX_LABELS, 5)  float  [cls, cx, cy, bw, bh] normalised
    counts   : (B,)  int  number of valid GT per image
    img_size : int   square input resolution

    Returns
    -------
    total_loss : scalar Tensor
    loss_dict  : {'box': float, 'dfl': float, 'cls': float}
    """
    B = labels.shape[0]

    # ── Build anchor grid ──────────────────────────────────────────────
    anchor_points, stride_tensor = make_anchors(raw_list, img_size, device)
    # anchor_points: (A, 2) pixel,  stride_tensor: (A, 1)

    # ── Decode predictions ────────────────────────────────────────────
    pred_bboxes, pred_scores, pred_dist_raw = decode_preds(
        raw_list, img_size, anchor_points, stride_tensor
    )
    # pred_bboxes   : (B, A, 4) xyxy pixels
    # pred_scores   : (B, A, NC) raw logits
    # pred_dist_raw : (B, A, 4*REG_MAX)

    # ── Prepare GT tensors (normalised → pixel xyxy) ──────────────────
    # Input: [cls, cx, cy, bw, bh] normalised → pixel xyxy
    gt_labels_pad = torch.zeros(B, labels.shape[1],          dtype=torch.long,  device=device)
    gt_bboxes_pad = torch.zeros(B, labels.shape[1], 4,       dtype=torch.float, device=device)
    mask_gt       = torch.zeros(B, labels.shape[1],          dtype=torch.bool,  device=device)

    for b in range(B):
        n = counts[b].item()
        if n == 0:
            continue
        gt = labels[b, :n]                 # (n, 5) [cls, cx, cy, bw, bh]
        gt_labels_pad[b, :n] = gt[:, 0].long()

        # cxcywh normalised → xyxy pixel
        cx = gt[:, 1] * img_size;  cy = gt[:, 2] * img_size
        bw = gt[:, 3] * img_size;  bh = gt[:, 4] * img_size
        gt_bboxes_pad[b, :n] = torch.stack(
            [cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2], dim=1
        )
        mask_gt[b, :n] = True

    # ── Task-Aligned Assignment ───────────────────────────────────────
    assigner = get_assigner(device)
    with torch.no_grad():
        (target_labels,   # (B, A)
         target_bboxes,   # (B, A, 4) pixel xyxy
         target_scores,   # (B, A, NC) soft iou-weighted one-hot
         fg_mask,         # (B, A) bool
         _) = assigner(
            pred_scores.sigmoid().detach(),
            pred_bboxes.detach(),
            anchor_points,
            gt_labels_pad,
            gt_bboxes_pad,
            mask_gt,
        )

    target_scores_sum = target_scores.sum().clamp(min=1.0)

    # ── Classification loss  (BCE with soft iou target) ───────────────
    # Exactly: bce(pred_scores, target_scores).sum() / target_scores_sum
    loss_cls = F.binary_cross_entropy_with_logits(
        pred_scores, target_scores.to(pred_scores.dtype), reduction="none"
    ).sum() / target_scores_sum

    # ── Box + DFL loss (foreground only) ─────────────────────────────
    loss_box = torch.tensor(0.0, device=device)
    loss_dfl = torch.tensor(0.0, device=device)

    n_fg = fg_mask.sum()
    if n_fg > 0:
        # weight per positive = normalised align score (target_scores for GT class)
        fg_pred_boxes = pred_bboxes[fg_mask]       # (P, 4) xyxy pixel
        fg_tgt_boxes  = target_bboxes[fg_mask]     # (P, 4) xyxy pixel
        fg_tgt_labels = target_labels[fg_mask]     # (P,)

        # IoU-based weight (sum-normalised, same as ultralytics)
        fg_iou = (1.0 - ciou_elementwise(fg_pred_boxes, fg_tgt_boxes)).detach()
        # or equivalently use target_scores class weight:
        weight = target_scores[fg_mask].gather(
            1, fg_tgt_labels.clamp(0, NUM_CLASSES - 1).unsqueeze(1)
        ).squeeze(1)                               # (P,)
        weight_sum = weight.sum().clamp(min=1.0)

        # CIoU loss
        ciou_vals = ciou_elementwise(fg_pred_boxes, fg_tgt_boxes)   # (P,)
        loss_box  = (ciou_vals * weight).sum() / weight_sum

        # DFL loss:  target = ltrb distances in stride-normalised bins
        fg_strides  = stride_tensor[fg_mask.view(-1)].squeeze(1)    # (P,)
        # recompute fg anchor_points
        fg_anc = anchor_points[fg_mask.view(-1)]                     # (P, 2)
        # target ltrb in pixels → divide by stride → in grid units (≈ bin index)
        tgt_ltrb = torch.stack([
            fg_anc[:, 0] - fg_tgt_boxes[:, 0],    # left
            fg_anc[:, 1] - fg_tgt_boxes[:, 1],    # top
            fg_tgt_boxes[:, 2] - fg_anc[:, 0],    # right
            fg_tgt_boxes[:, 3] - fg_anc[:, 1],    # bottom
        ], dim=1) / fg_strides[:, None]            # (P, 4)

        fg_pred_dist = pred_dist_raw[fg_mask].view(-1, 4, REG_MAX)  # (P, 4, REG_MAX)
        loss_dfl = dfl_loss(fg_pred_dist, tgt_ltrb)

    total_loss = (LAMBDA_BOX * loss_box +
                  LAMBDA_DFL * loss_dfl +
                  LAMBDA_CLS * loss_cls)

    return total_loss, {
        "box": loss_box.item(),
        "dfl": loss_dfl.item(),
        "cls": loss_cls.item(),
    }


# ===========================================================================
# ④ Post-processing (inference → eval format)
# ===========================================================================

def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float) -> torch.Tensor:
    """Greedy NMS. boxes: (N,4) xyxy, scores: (N,). Returns kept indices."""
    order = scores.argsort(descending=True)
    keep  = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        from prepare import box_iou_batch, cxcywh_to_xyxy
        iou  = box_iou_batch(boxes[i:i+1], boxes[rest])[0]
        order = rest[iou <= iou_thr]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def postprocess(raw_list: list[torch.Tensor],
                img_size: int,
                conf_thr: float = CONF_THRESHOLD,
                nms_thr:  float = NMS_IOU_THR):
    """
    Convert raw head outputs to final detection tensors.

    Returns list of Tensors, one per image:
        (P, 6)  cx, cy, w, h, conf, cls   — normalised [0,1]
    """
    from prepare import cxcywh_to_xyxy
    boxes, logits, _, _ = decode_preds(raw_list, img_size)
    # boxes  : (B, A, 4)
    # logits : (B, A, NC)
    B     = boxes.shape[0]
    confs = logits.sigmoid()    # (B, A, NC)

    results = []
    for b in range(B):
        scores, cls_ids = confs[b].max(dim=-1)     # (A,), (A,)
        mask = scores > conf_thr

        if not mask.any():
            results.append(torch.zeros((0, 6), device=boxes.device))
            continue

        b_boxes  = boxes[b][mask]    # (P, 4)  cxcywh
        b_scores = scores[mask]      # (P,)
        b_cls    = cls_ids[mask].float()

        # NMS per class
        kept = []
        for c in range(NUM_CLASSES):
            cmask = b_cls == c
            if not cmask.any():
                continue
            cb   = b_boxes[cmask]
            cs   = b_scores[cmask]
            keep = nms(cxcywh_to_xyxy(cb), cs, nms_thr)
            kept.append(torch.cat([
                cb[keep],
                cs[keep].unsqueeze(1),
                b_cls[cmask][keep].unsqueeze(1),
            ], dim=1))

        if kept:
            results.append(torch.cat(kept, dim=0))
        else:
            results.append(torch.zeros((0, 6), device=boxes.device))

    return results


# ===========================================================================
# ⑤ Training loop
# ===========================================================================

def main():
    program_start = time.perf_counter()   # wall-clock start of entire run

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Dataloaders ──────────────────────────────────────────────────────
    train_loader = get_dataloader(
        TRAIN_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, augment=True, shuffle=True,
    )
    val_loader = get_dataloader(
        VAL_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, augment=False, shuffle=False,
    )
    print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = YOLOv12().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params / 1e6:.2f}M")

    # ── Optimiser ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    # Cosine LR with linear warm-up
    def lr_lambda(step: int) -> float:
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        # cosine decay from 1 → 0.01
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return 0.01 + 0.5 * 0.99 * (1 + math.cos(math.pi * progress))

    # We need total_steps for the scheduler — estimate from TIME_BUDGET
    # (rough: 1 epoch ≈ len(train_loader) steps, but time-budgeted)
    total_steps = 10_000   # generous upper bound; scheduler will clamp

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Loss history ──────────────────────────────────────────────────────
    import csv
    loss_log: list[dict] = []

    # ── Training ──────────────────────────────────────────────────────────
    scaler     = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    step       = 0
    train_start = time.perf_counter()
    deadline    = train_start + TIME_BUDGET

    model.train()
    epoch = 0
    while True:
        epoch += 1
        for imgs, labels, counts in train_loader:
            if time.perf_counter() >= deadline:
                break

            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            counts = counts.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type,
                                    enabled=(device.type == "cuda")):
                raw_list = model(imgs)
                loss, loss_dict = compute_loss(raw_list, labels, counts, IMG_SIZE, device)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1

            # ── Record losses every step ───────────────────────────────
            elapsed = time.perf_counter() - train_start
            lr_now  = scheduler.get_last_lr()[0]
            loss_log.append({
                "step":    step,
                "epoch":   epoch,
                "total":   round(loss.item(),          6),
                "ciou":    round(loss_dict["box"],     6),
                "dfl":     round(loss_dict["dfl"],     6),
                "bce":     round(loss_dict["cls"],     6),
                "lr":      round(lr_now,               8),
                "elapsed": round(elapsed,              2),
            })

            if step % 100 == 0:
                remaining = max(0, TIME_BUDGET - elapsed)
                print(f"  step={step:5d}  total={loss.item():.4f}  "
                      f"ciou={loss_dict['box']:.3f}  dfl={loss_dict['dfl']:.3f}  "
                      f"bce={loss_dict['cls']:.3f}  "
                      f"lr={lr_now:.2e}  elapsed={elapsed:.0f}s  remaining={remaining:.0f}s")

        if time.perf_counter() >= deadline:
            break

    elapsed_train = time.perf_counter() - train_start
    print(f"\nTraining finished: {step} steps, {epoch} epochs, {elapsed_train:.1f}s")

    # ── Save loss history → CSV ────────────────────────────────────────────
    loss_csv = Path("loss_history.csv")
    with open(loss_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["step", "epoch", "total", "ciou", "dfl", "bce",
                           "lr", "elapsed"]
        )
        writer.writeheader()
        writer.writerows(loss_log)
    print(f"Loss history saved → {loss_csv}  ({len(loss_log)} rows)")

    # ── Evaluation helper ────────────────────────────────────────────────
    def run_eval(loader):
        preds, gts = [], []
        with torch.no_grad():
            for imgs, labels, counts in loader:
                imgs     = imgs.to(device, non_blocking=True)
                raw_list = model(imgs)
                det_list = postprocess(raw_list, IMG_SIZE)
                for b in range(len(imgs)):
                    n = counts[b].item()
                    preds.append(det_list[b].cpu())
                    gts.append(labels[b, :n].cpu())
        return evaluate(preds, gts)

    model.eval()

    # Val split
    metrics = run_eval(val_loader)

    # Test-dev split (if available)
    test_metrics = None
    if TEST_DIR.exists() and (TEST_DIR / "labels").exists():
        test_loader = get_dataloader(
            TEST_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS, augment=False, shuffle=False,
        )
        test_metrics = run_eval(test_loader)

    # Testset-challenge split (if available)
    challenge_metrics = None
    if CHALLENGE_DIR.exists() and (CHALLENGE_DIR / "labels").exists():
        challenge_loader = get_dataloader(
            CHALLENGE_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS, augment=False, shuffle=False,
        )
        challenge_metrics = run_eval(challenge_loader)

    # Peak VRAM
    peak_vram_mb = 0
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device) // (1024 * 1024)

    # ── Summary stats ─────────────────────────────────────────────────────
    run_end        = time.perf_counter()
    total_seconds  = run_end - program_start   # includes model build + eval
    num_params     = sum(p.numel() for p in model.parameters()) / 1e6

    # MFU: approximate (6 * params * batch_size) flops per step
    mfu_percent = 0.0
    if device.type == "cuda" and step > 0:
        try:
            gpu_flops_per_sec = {
                "H100":  989e12,  "A100":  312e12,
                "A6000": 309e12,  "RTX 4090": 82.6e12,
                "RTX 3090": 35.6e12,
            }
            gpu_name = torch.cuda.get_device_name(device)
            peak_flops = next(
                (v for k, v in gpu_flops_per_sec.items() if k in gpu_name), None
            )
            if peak_flops:
                flops_per_step = 6 * (num_params * 1e6) * BATCH_SIZE
                total_flops    = flops_per_step * step
                mfu_percent    = 100.0 * total_flops / (peak_flops * elapsed_train)
        except Exception:
            pass

    # ── Results  (grep-friendly format used by autoresearch harness) ──────
    print(f"\nval_box_iou:       {metrics['val_box_iou']:.4f}")
    print(f"val_cls_acc:       {metrics['val_cls_acc']:.4f}")
    if test_metrics is not None:
        print(f"test_box_iou:      {test_metrics['val_box_iou']:.4f}")
        print(f"test_cls_acc:      {test_metrics['val_cls_acc']:.4f}")
    if challenge_metrics is not None:
        print(f"challenge_test_box_iou:  {challenge_metrics['val_box_iou']:.4f}")
        print(f"challenge_test_cls_acc:  {challenge_metrics['val_cls_acc']:.4f}")
    print(f"training_seconds:  {elapsed_train:.1f}")
    print(f"total_seconds:     {total_seconds:.1f}")
    print(f"peak_vram_mb:      {peak_vram_mb}")
    print(f"mfu_percent:       {mfu_percent:.2f}")
    print(f"num_steps:         {step}")
    print(f"num_params_M:      {num_params:.1f}")


if __name__ == "__main__":
    main()
