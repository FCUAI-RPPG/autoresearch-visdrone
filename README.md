# autoresearch-visdrone
!!! I am not test before, because this code is not done !!!

Autonomous fine-tuning research on **VisDrone Task-1 Object Detection** using **YOLOv12**, inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

An AI agent is given a pretrained YOLOv12s model and a fine-tuning script. It experiments autonomously — modifying code, training for 5 minutes, checking if results improved, keeping or discarding, and repeating. You wake up to a log of experiments and (hopefully) a better model.

---

## Repository structure

```
prepare.py          — fixed. Data prep, dataloader, anchor analysis, evaluation. Do not modify.
train_simple.py     — fine-tune YOLOv12s via ultralytics. Agent modifies this.
train.py            — full YOLOv12 from scratch with custom loss. Agent modifies this.
program.md          — agent instructions for train_simple.py experiments.
pyproject.toml      — dependencies (managed by uv).
```

---

## How it works

```
yolov12s.pt  (COCO pretrained)
      │
      ├─ first FREEZE_LAYERS frozen   ← backbone stays fixed
      │
      └─ unfrozen layers + head       ← fine-tuned on VisDrone
```

Every experiment runs for a fixed **5-minute wall-clock budget**. The agent edits `train_simple.py`, runs it, reads two metrics, and keeps or reverts:

| Metric | Description | Direction |
|---|---|---|
| `val_box_iou` | Mean best-GT IoU of all predictions | higher is better |
| `val_cls_acc` | Classification accuracy of matched pred–GT pairs | higher is better |

---

## Dataset — VisDrone Task-1

Download from [aiskyeye.com](http://aiskyeye.com) (registration required). Expected directory layout:

```
<VISDRONE_ROOT>/
    VisDrone2019-DET-train/
        images/          *.jpg
        annotations/     *.txt
    VisDrone2019-DET-val/
        images/
        annotations/
    VisDrone2019-DET-test-dev/
        images/
        annotations/
    VisDrone2019-DET-testset-challenge/
        images/
        annotations/
```

**10 object categories:** pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor.

---

## Quick start

**Requirements:** single NVIDIA GPU, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and install dependencies
git clone https://github.com/FCUAI-RPPG/autoresearch-visdrone.git
cd autoresearch-visdrone
uv sync

# 3. Prepare dataset — one-time, converts VisDrone annotations to YOLO format
VISDRONE_ROOT=/path/to/visdrone uv run prepare.py --root /path/to/visdrone

# 4. Run a single training experiment (~5 min)
VISDRONE_ROOT=/path/to/visdrone uv run train_simple.py
```

---

## Running the agent

Point your Claude / Codex agent at this repo and prompt:

```
Have a look at program.md and let's kick off a new experiment. Let's do the setup first.
```

The agent reads `program.md`, creates a branch, establishes a baseline, and loops autonomously.

---

## Project files

### `prepare.py` — fixed, do not modify

One-time data preparation and runtime utilities:

- Converts VisDrone `.txt` annotations to YOLO-format label files (`labels/*.txt`)
- K-means anchor analysis on GT boxes (IoU-distance metric, YOLOv5-style)
- `VisDroneDetDataset` and `get_dataloader()` for training
- `evaluate()` returning `val_box_iou` and `val_cls_acc`

```bash
uv run prepare.py --root /path/to/visdrone [--imgsz 640] [--anchors 9]
```

### `train_simple.py` — agent modifies this

Fine-tunes the official `yolov12s.pt` pretrained on COCO:

- Loads weights via `ultralytics.YOLO`
- Freezes first `FREEZE_LAYERS` layers (default: 10)
- Replaces detection head for 10 VisDrone classes
- Uses `ultralytics.utils.loss.v8DetectionLoss` (TAL + CIoU + DFL + BCE)
- Saves `loss_history.csv` (per-step `total / ciou / dfl / bce`)

**Key hyperparameters** (all in `① Hyperparameters` section):

| Variable | Default | Description |
|---|---|---|
| `WEIGHTS` | `yolov12s.pt` | Pretrained weights |
| `FREEZE_LAYERS` | `10` | Layers to freeze from backbone |
| `LR` | `1e-3` | AdamW learning rate |
| `WEIGHT_DECAY` | `1e-4` | L2 regularisation |
| `WARMUP_STEPS` | `200` | Linear LR warm-up steps |
| `BATCH_SIZE` | `8` | Images per step |
| `CONF_THRESHOLD` | `0.25` | Confidence threshold at eval |
| `NMS_IOU_THR` | `0.45` | NMS IoU threshold at eval |
| `IMG_SIZE` | `640` | Model input resolution |

### `train.py` — full YOLOv12 from scratch

Custom YOLOv12 implementation with:

- CSPDarkNet backbone + PANet neck
- Anchor-free decoupled detection head with DFL
- Full `v8DetectionLoss` pipeline (TaskAlignedAssigner + CIoU + DFL + BCE)
- Cosine LR schedule with linear warm-up

### `program.md` — agent instructions

Defines the experiment loop, constraints, output format, and logging rules for the agent running `train_simple.py`. Read this before starting the agent.

---

## Output format

Each run prints at the end:

```
val_box_iou:              0.XXXX
val_cls_acc:              0.XXXX
test_box_iou:             0.XXXX   (if test-dev labels exist)
test_cls_acc:             0.XXXX
challenge_test_box_iou:   0.XXXX   (if testset-challenge labels exist)
challenge_test_cls_acc:   0.XXXX
training_seconds:         300.1
total_seconds:            325.9
peak_vram_mb:             6142
mfu_percent:              39.80
num_steps:                953
num_params_M:             50.3
```

Extract key metrics from the log:

```bash
grep "^val_box_iou:\|^val_cls_acc:\|^peak_vram_mb:" run.log
```

---

## Experiment results

Results are logged in `results.tsv` (tab-separated, untracked by git):

```
commit	val_box_iou	val_cls_acc	memory_gb	status	description
a1b2c3d	0.4210	0.7830	6.0	keep	baseline FREEZE_LAYERS=10 LR=1e-3
b2c3d4e	0.4380	0.7910	6.1	keep	LR=5e-4 WARMUP_STEPS=500
c3d4e5f	0.4190	0.7750	6.0	discard	LR=2e-3 unstable loss
d4e5f6g	0.0000	0.0000	0.0	crash	BATCH_SIZE=32 OOM
```

---

## License

MIT
