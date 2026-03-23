"""
train_simple.py — Fine-tune YOLOv12s on VisDrone Task-1
                  using Ultralytics as the backbone.

Strategy:
  1. Load official yolov12s.pt pretrained weights
  2. Freeze the first FREEZE_LAYERS layers (default: 10)
  3. Fine-tune unfrozen layers + head on VisDrone
  4. Evaluate with prepare.py metrics (val_box_iou, val_cls_acc)

Metrics printed at end (grep-friendly):
    val_box_iou: 0.XXXX
    val_cls_acc: 0.XXXX
    peak_vram_mb: XXXX
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from ultralytics import YOLO

from prepare import (
    NUM_CLASSES,
    evaluate,
    get_dataloader,
)

# ===========================================================================
# ① Hyperparameters — agent tweaks these
# ===========================================================================

VISDRONE_ROOT  = Path(os.environ.get("VISDRONE_ROOT", "/data/visdrone"))
TRAIN_DIR      = VISDRONE_ROOT / "VisDrone2019-DET-train"
VAL_DIR        = VISDRONE_ROOT / "VisDrone2019-DET-val"
TEST_DIR       = VISDRONE_ROOT / "VisDrone2019-DET-test-dev"
CHALLENGE_DIR  = VISDRONE_ROOT / "VisDrone2019-DET-testset-challenge"

# ── Pretrained weights ──────────────────────────────────────────────────────
WEIGHTS        = "yolov12s.pt"      # downloaded automatically by ultralytics

# ── Freeze: first N layers of model.model are frozen ───────────────────────
FREEZE_LAYERS  = 10                 # 0 = train everything, 10 = freeze backbone

# ── Training ────────────────────────────────────────────────────────────────
IMG_SIZE       = 640
BATCH_SIZE     = 8
NUM_WORKERS    = 4
TIME_BUDGET    = 300                # wall-clock training seconds (5 min)

LR             = 1e-3               # initial lr for unfrozen params (AdamW)
LR_FROZEN_HEAD = 1e-3               # lr for detection head (always trainable)
WEIGHT_DECAY   = 1e-4
WARMUP_STEPS   = 200
MAX_GRAD_NORM  = 10.0

# ── Inference ───────────────────────────────────────────────────────────────
CONF_THRESHOLD = 0.25
NMS_IOU_THR    = 0.45


# ===========================================================================
# ── Model Setup ──
# ===========================================================================

def build_model(weights: str, freeze_layers: int, num_classes: int, device):
    """
    Load YOLOv12s pretrained weights via ultralytics,
    replace the detection head for num_classes,
    and freeze the first freeze_layers layers.

    Returns the raw nn.Module (model.model) ready for manual training.
    """
    # Load via ultralytics YOLO wrapper
    yolo = YOLO(weights)

    # Access the underlying nn.Module
    net = yolo.model                    # ultralytics DetectionModel (nn.Module)
    net = net.to(device)

    # ── Replace head output channels for num_classes ─────────────────────
    # The detection head is the last module in net.model
    head = net.model[-1]               # Detect module
    if hasattr(head, "nc") and head.nc != num_classes:
        print(f"  Replacing head: {head.nc} classes → {num_classes} classes")
        head.nc = num_classes
        # Re-initialise the class prediction convolutions
        for cv2 in head.cv2:           # box regression branch  (no change needed)
            pass
        for i, cv3 in enumerate(head.cv3):   # class prediction branch
            # cv3 is Sequential: [Conv, Conv, Conv2d]
            last = cv3[-1]
            in_ch = last.in_channels
            # Replace final conv
            new_conv = nn.Conv2d(in_ch, num_classes, 1)
            nn.init.kaiming_normal_(new_conv.weight, nonlinearity="relu")
            # Bias: prior prob = 0.01
            import math
            nn.init.constant_(new_conv.bias, math.log(0.01 / 0.99))
            cv3[-1] = new_conv.to(device)
        head.no = num_classes + head.reg_max * 4   # outputs per anchor

    # ── Freeze first freeze_layers layers ────────────────────────────────
    # net.model is an nn.Sequential of backbone + neck + head blocks
    frozen_count = 0
    for i, layer in enumerate(net.model):
        if i < freeze_layers:
            for param in layer.parameters():
                param.requires_grad_(False)
            frozen_count += sum(p.numel() for p in layer.parameters())
        else:
            for param in layer.parameters():
                param.requires_grad_(True)

    total      = sum(p.numel() for p in net.parameters())
    trainable  = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"  Layers frozen   : {freeze_layers} / {len(list(net.model))}")
    print(f"  Frozen params   : {frozen_count / 1e6:.2f}M")
    print(f"  Trainable params: {trainable / 1e6:.2f}M  /  {total / 1e6:.2f}M total")

    return net


def build_optimizer(net: nn.Module, lr: float, weight_decay: float):
    """AdamW with separate param groups: frozen layers excluded."""
    trainable = [p for p in net.parameters() if p.requires_grad]
    return torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)


# ---------------------------------------------------------------------------
# Post-processing: ultralytics raw output → prepare.evaluate() format
# ---------------------------------------------------------------------------

def postprocess_ultralytics(raw, conf_thr: float, nms_thr: float, device):
    """
    Convert ultralytics model output to the format expected by prepare.evaluate():
        list of Tensors, each (P, 6): cx, cy, w, h, conf, cls  (normalised [0,1])

    ultralytics Detect head returns either:
      - tuple(pred, ...)  where pred is list of (B, no, H, W)  in training mode
      - Tensor (B, no, A) in eval mode  (after internal decode)

    We call net in train mode so we get raw tensors, then decode manually
    using the ultralytics built-in decode helper.
    """
    # In eval mode ultralytics returns (B, 4+NC, A) after decoding
    # raw is already decoded boxes+scores tensor: (B, 4+NC, A)
    B = raw.shape[0]
    results = []

    for b in range(B):
        pred = raw[b].T                        # (A, 4+NC)  cx cy w h cls...
        boxes  = pred[:, :4]                   # (A, 4)  cx cy w h  (pixel or norm?)
        scores = pred[:, 4:]                   # (A, NC)

        conf, cls_ids = scores.max(dim=1)      # (A,), (A,)
        mask = conf > conf_thr
        if not mask.any():
            results.append(torch.zeros((0, 6), device=device))
            continue

        b_boxes  = boxes[mask]
        b_conf   = conf[mask]
        b_cls    = cls_ids[mask].float()

        # Per-class NMS
        kept = []
        for c in range(NUM_CLASSES):
            cm = b_cls == c
            if not cm.any():
                continue
            cb = b_boxes[cm]
            cs = b_conf[cm]
            # torchvision NMS
            try:
                from torchvision.ops import nms as tv_nms
                from prepare import cxcywh_to_xyxy
                keep = tv_nms(cxcywh_to_xyxy(cb), cs, nms_thr)
            except Exception:
                keep = torch.arange(len(cb), device=device)
            kept.append(torch.cat([
                cb[keep],
                cs[keep].unsqueeze(1),
                b_cls[cm][keep].unsqueeze(1),
            ], dim=1))

        if kept:
            results.append(torch.cat(kept, dim=0))
        else:
            results.append(torch.zeros((0, 6), device=device))

    return results


# ===========================================================================
# ② Training loop
# ===========================================================================

def main():
    program_start = time.perf_counter()   # wall-clock start of entire run

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

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
    print(f"\nLoading {WEIGHTS}  (freeze first {FREEZE_LAYERS} layers)")
    net = build_model(WEIGHTS, FREEZE_LAYERS, NUM_CLASSES, device)

    # ── Optimiser & scheduler ──────────────────────────────────────────
    optimizer = build_optimizer(net, LR, WEIGHT_DECAY)
    total_steps = 10_000   # upper bound for cosine schedule

    def lr_lambda(step: int) -> float:
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return 0.01 + 0.5 * 0.99 * (1 + __import__("math").cos(
            __import__("math").pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    # ── Use ultralytics built-in loss ─────────────────────────────────
    # v8DetectionLoss returns: total_loss, loss_items[box_ciou, cls_bce, dfl]
    from ultralytics.utils.loss import v8DetectionLoss
    from types import SimpleNamespace

    # Patch model.args so v8DetectionLoss initialises correctly
    net.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

    criterion = v8DetectionLoss(net)

    # Also patch criterion.hyp directly in case ultralytics stored a dict
    if isinstance(criterion.hyp, dict):
        criterion.hyp = SimpleNamespace(**criterion.hyp)
    if not hasattr(criterion.hyp, "box"):
        criterion.hyp.box = 7.5
    if not hasattr(criterion.hyp, "cls"):
        criterion.hyp.cls = 0.5
    if not hasattr(criterion.hyp, "dfl"):
        criterion.hyp.dfl = 1.5

    # ── Loss history (one dict per step, saved to CSV at end) ────────
    import csv
    loss_log: list[dict] = []

    def _parse_loss_items(loss_items):
        """Unpack ultralytics loss_items Tensor([ciou, bce, dfl]) safely."""
        if hasattr(loss_items, "tolist"):
            vals = loss_items.tolist()
            ciou = vals[0] if len(vals) > 0 else 0.0
            bce  = vals[1] if len(vals) > 1 else 0.0
            dfl  = vals[2] if len(vals) > 2 else 0.0
        else:
            ciou = bce = dfl = 0.0
        return ciou, bce, dfl

    # ── Training ──────────────────────────────────────────────────────
    net.train()
    step        = 0
    train_start = time.perf_counter()
    deadline    = train_start + TIME_BUDGET
    epoch       = 0

    while True:
        epoch += 1
        for imgs, labels, counts in train_loader:
            if time.perf_counter() >= deadline:
                break

            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            counts = counts.to(device, non_blocking=True)

            # ── Convert labels to ultralytics batch format ────────────
            batch_idx_list, cls_list, bbox_list = [], [], []
            for b in range(len(imgs)):
                n = counts[b].item()
                if n == 0:
                    continue
                gt = labels[b, :n]
                batch_idx_list.append(torch.full((n,), b, device=device))
                cls_list.append(gt[:, 0])
                bbox_list.append(gt[:, 1:])

            ul_batch = {
                "batch_idx": torch.cat(batch_idx_list) if batch_idx_list
                             else torch.zeros(0, device=device),
                "cls":       torch.cat(cls_list).unsqueeze(1) if cls_list
                             else torch.zeros((0, 1), device=device),
                "bboxes":    torch.cat(bbox_list) if bbox_list
                             else torch.zeros((0, 4), device=device),
                "img":       imgs,
            }

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type,
                                    enabled=(device.type == "cuda")):
                preds = net(imgs)
                loss, loss_items = criterion(preds, ul_batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in net.parameters() if p.requires_grad],
                MAX_GRAD_NORM,
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1

            # ── Record losses ─────────────────────────────────────────
            elapsed = time.perf_counter() - train_start
            lr_now  = scheduler.get_last_lr()[0]
            ciou, bce, dfl = _parse_loss_items(loss_items)
            total = loss.item()

            loss_log.append({
                "step":    step,
                "epoch":   epoch,
                "total":   round(total, 6),
                "ciou":    round(ciou,  6),
                "dfl":     round(dfl,   6),
                "bce":     round(bce,   6),
                "lr":      round(lr_now, 8),
                "elapsed": round(elapsed, 2),
            })

            if step % 50 == 0:
                remaining = max(0, TIME_BUDGET - elapsed)
                print(f"  step={step:4d}  total={total:.4f}  "
                      f"ciou={ciou:.3f}  dfl={dfl:.3f}  bce={bce:.3f}  "
                      f"lr={lr_now:.2e}  {elapsed:.0f}s/{TIME_BUDGET}s")

        if time.perf_counter() >= deadline:
            break

    elapsed_train = time.perf_counter() - train_start
    print(f"\nTraining done: {step} steps, {epoch} epochs, {elapsed_train:.1f}s")

    # ── Save loss history → CSV ───────────────────────────────────────
    loss_csv = Path("loss_history.csv")
    with open(loss_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["step", "epoch", "total", "ciou", "dfl", "bce",
                           "lr", "elapsed"]
        )
        writer.writeheader()
        writer.writerows(loss_log)
    print(f"Loss history saved → {loss_csv}  ({len(loss_log)} rows)")

    # ── Evaluation helper ────────────────────────────────────────────
    def run_eval(loader):
        preds, gts = [], []
        with torch.no_grad():
            for imgs, labels, counts in loader:
                imgs = imgs.to(device, non_blocking=True)
                raw = net(imgs)
                if isinstance(raw, (list, tuple)):
                    raw = raw[0]
                if raw.shape[1] >= 4:
                    raw = raw.clone()
                    raw[:, :4] /= IMG_SIZE
                det_list = postprocess_ultralytics(raw, CONF_THRESHOLD,
                                                   NMS_IOU_THR, device)
                for b in range(len(imgs)):
                    n = counts[b].item()
                    preds.append(det_list[b].cpu())
                    gts.append(labels[b, :n].cpu())
        return evaluate(preds, gts)

    net.eval()

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

    peak_vram_mb = 0
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device) // (1024 * 1024)

    # ── Summary stats ─────────────────────────────────────────────────
    run_end        = time.perf_counter()
    total_seconds  = run_end - program_start   # includes model load + eval
    num_params     = sum(p.numel() for p in net.parameters()) / 1e6

    # MFU: (flops per step) / (peak_flops_per_sec * training_seconds)
    # Approximate FLOPs per forward+backward as 6 * num_params * BATCH_SIZE * IMG_SIZE^2
    # This is a rough proxy; real FLOPs depend on architecture details.
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

    # ── Results  (grep-friendly, used by autoresearch harness) ────────
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
