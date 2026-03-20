# Program — VisDrone YOLOv12s Fine-tuning

You are an autonomous research agent. Your job is to fine-tune **YOLOv12s** on the **VisDrone Task-1** object detection dataset by iterating on `train_simple.py`, running experiments, and keeping only the changes that improve the metrics. You run experiments back-to-back without stopping until the human interrupts you.

The repo has exactly three files that matter:
- `prepare.py` — fixed. Data prep, dataloader, evaluation utilities. Do not touch.
- `train_simple.py` — the file you modify. Pretrained YOLOv12s backbone, frozen layers, optimizer, fine-tuning loop. Only the `① Hyperparameters` section at the top is in scope; everything below `── Model Setup ──` is off-limits.
- `program.md` — this file. Instructions for you.

---

## Setup

Work with the user to complete the following before the experiment loop begins.

1. **Agree on a run tag.** Propose a tag based on today's date (e.g. `mar20`). The branch `autoresearch_yolov12/<tag>` must not already exist — this is a fresh run.
2. **Create the branch.**
   ```bash
   git checkout -b autoresearch_yolov12/<tag>
   ```
3. **Read the in-scope files.** The repo is small. Read all of these before touching anything:
   - `program.md` — this file.
   - `train_simple.py` — the file you modify. Pretrained YOLOv12s backbone, frozen layers, optimizer, fine-tuning loop. Read the `① Hyperparameters` section carefully. Note which variables exist, their defaults, and the constraint that nothing below `── Model Setup ──` may change.
   - `prepare.py` — read for context on `get_dataloader`, `evaluate`, `NUM_CLASSES`. Do not modify.
4. **Verify data exists.** Check that `$VISDRONE_ROOT/VisDrone2019-DET-train/labels/` is populated. If not, tell the human to run:
   ```bash
   uv run prepare.py --root /path/to/visdrone
   ```
5. **Initialise `results.tsv`.** Create the file with only the header row. The baseline will be recorded after the first run.
6. **Confirm and go.** Once setup looks good, kick off the first experiment immediately.

---

## Experimentation

Each experiment runs on a single GPU. The training script runs for a fixed time budget of 5 minutes (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train_simple.py`.

### What you CAN do:

- Modify `train_simple.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

### What you CANNOT do:

- Modify `prepare.py` in any way.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness.
- Change `TIME_BUDGET` — every run must train for exactly 5 minutes.

The goal is simple: get the highest `val_box_iou` and `val_cls_acc`. Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the freeze strategy, the optimizer, the hyperparameters, the batch size, the pretrained weights. The only constraint is that the code runs without crashing and finishes within the time budget.

VRAM is a soft constraint. Some increase is acceptable for meaningful metric gains, but it should not blow up dramatically.

Simplicity criterion: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

The first run: Your very first run should always be to establish the baseline, so you will run the training script as is.

---

## Output format

Once the script finishes it prints a summary like this:

```
val_box_iou:       0.XXXX
val_cls_acc:       0.XXXX
training_seconds:  300.1
total_seconds:     325.9
peak_vram_mb:      6142
mfu_percent:       39.80
num_steps:         953
num_params_M:      50.3
```

Note that the script is configured to always stop after 5 minutes, so depending on the computing platform the numbers might look different. You can extract the key metrics from the log file:

```bash
grep "^val_box_iou:\|^val_cls_acc:\|^peak_vram_mb:" run.log
```

The run also saves `loss_history.csv` — a per-step log of `total / ciou / dfl / bce` losses. You do not need to read this for the keep/discard decision, but it is useful for diagnosing instability (e.g. a spike in `dfl` or a flat `bce` suggests a specific problem).

Both `val_box_iou` and `val_cls_acc` are in `[0, 1]`. **Higher is better for both.**

---

## Logging results

Record every completed run (including crashes) in `results.tsv` using tab-separated values. Do not use commas — they break descriptions. The TSV has a header row and 6 columns:

1. `commit` — git commit hash (short, 7 chars). Use `git log --oneline -1` to get it after a keep. For discard/crash, use the hash of the current HEAD.
2. `val_box_iou` — achieved value (e.g. `0.421000`) — use `0.000000` for crashes.
3. `val_cls_acc` — achieved value (e.g. `0.783000`) — use `0.000000` for crashes.
4. `memory_gb` — peak memory in GB, rounded to `.1f` (divide `peak_vram_mb` by 1024, e.g. `6142 / 1024 = 6.0`) — use `0.0` for crashes.
5. `status` — one of: `keep`, `discard`, `crash`.
6. `description` — short text description of what this experiment tried.

Example:

```
commit	val_box_iou	val_cls_acc	memory_gb	status	description
a1b2c3d	0.4210	0.7830	6.0	keep	baseline FREEZE_LAYERS=10 LR=1e-3
b2c3d4e	0.4380	0.7910	6.1	keep	LR=5e-4 WARMUP_STEPS=500
c3d4e5f	0.4190	0.7750	6.0	discard	LR=2e-3 unstable loss
d4e5f6g	0.0000	0.0000	0.0	crash	BATCH_SIZE=32 OOM
```

---

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch_yolov12/mar20` or `autoresearch_yolov12/mar20-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Tune `train_simple.py` with an experimental idea by directly hacking the code.
3. `git commit`
4. Run the experiment:
   ```bash
   uv run train_simple.py > run.log 2>&1
   ```
   (redirect everything — do NOT use `tee` or let output flood your context)
5. Read out the results:
   ```bash
   grep "^val_box_iou:\|^val_cls_acc:\|^peak_vram_mb:" run.log
   ```
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the `results.tsv` file, leave it untracked by git).
8. If `val_box_iou` or `val_cls_acc` improved (higher), you "advance" the branch, keeping the git commit.
9. If both metrics are equal or worse, you `git reset` back to where you started.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

Timeout: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

Crashes: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log `crash` as the status in the tsv, and move on.

NEVER STOP: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working indefinitely until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical hyperparameter changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

