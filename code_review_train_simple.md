# Code Review: train_simple.py

審查日期：2026-03-27
對象：`train_simple.py` — Fine-tune YOLOv12s on VisDrone Task-1

---

## 已修正

### MFU 公式少乘 `IMG_SIZE²`
- **位置**：line 439
- **問題**：FLOPs per step 缺少空間解析度因子，導致 MFU 嚴重低估（差距達 640² = 409,600 倍）
- **修正**：
  ```python
  # Before
  flops_per_step = 6 * (num_params * 1e6) * BATCH_SIZE
  # After
  flops_per_step = 6 * (num_params * 1e6) * BATCH_SIZE * IMG_SIZE ** 2
  ```

### Eval output 解包方式脆弱
- **位置**：`run_eval()` line 378-382
- **問題一**：`raw = raw[0]` 盲目取 tuple 第一個元素，不同 ultralytics 版本第一個元素不一定是 decoded pred tensor
- **問題二**：`raw[:, :4] /= IMG_SIZE` 假設圖為正方形，letterbox 情況下 x/y 和 w/h 偏移不同
- **修正**：
  ```python
  # 找 shape == (B, 4+NC, A) 的 tensor
  if isinstance(raw, (list, tuple)):
      decoded = None
      for item in raw:
          if (isinstance(item, torch.Tensor)
                  and item.ndim == 3
                  and item.shape[1] == 4 + NUM_CLASSES):
              decoded = item
              break
      raw = decoded if decoded is not None else raw[0]

  # 分軸 normalize
  _, _, h, w = imgs.shape
  raw = raw.clone()
  raw[:, 0] /= w   # cx
  raw[:, 1] /= h   # cy
  raw[:, 2] /= w   # bw
  raw[:, 3] /= h   # bh
  ```

---

## 待修正

### 🔴 NMS 失敗時靜默吞掉錯誤
- **位置**：`postprocess_ultralytics()` line 185-186
- **問題**：`except Exception` 捕捉所有錯誤後 fallback 為「保留全部 box 不做 NMS」，mAP 會因大量重複框暴跌，但程式不報錯，極難察覺
- **建議**：
  ```python
  except Exception as e:
      print(f"[WARN] NMS failed for class {c}: {e}, skipping NMS")
      keep = torch.arange(len(cb), device=device)
  ```

### 🔴 `hyp` 讀值用 `or` 有 0 值陷阱
- **位置**：line 252-258
- **問題**：Python 的 `or` 遇到 `0` / `0.0` 也會視為 falsy 繼續往後，若 ultralytics 某版本把 hyp 值設為 `0`，會誤用 hardcode fallback
- **建議**：
  ```python
  hyp_box = getattr(criterion.hyp, "box", None)
  if hyp_box is None and isinstance(criterion.hyp, dict):
      hyp_box = criterion.hyp.get("box")
  if hyp_box is None:
      hyp_box = 7.5
  ```

### ~~🟡 `head.stride` 未初始化可能導致 forward crash~~（不適用，可忽略）
- **位置**：`build_model()` line 90-107
- **說明**：`stride` 由網路架構的 downsampling 倍率決定，載入 `yolov12s.pt` 時已初始化完畢。換 `head.nc` / `cv3[-1]` 只改 conv output channels，不影響 stride。僅在從頭建模（無預訓練權重）時才需要 dummy forward。

### ~~🟡 `scores` 假設已是 sigmoid 輸出，沒有保護~~（不適用，可忽略）
- **位置**：`postprocess_ultralytics()` line 162-163
- **問題**：eval mode 輸出的 scores 是 sigmoid 後的機率值，`> 0.25` 語意正確；但若換成 train mode 的 raw logits，logit=0.25 對應機率約 0.56，閾值語意完全不同，且不會報錯
- **建議**：加上斷言或說明，或在函式入口明確 sigmoid：
  ```python
  # 確保 scores 在 [0,1] 範圍
  assert scores.min() >= 0 and scores.max() <= 1, \
      "scores should be sigmoid probabilities, not raw logits"
  ```

### 🟡 `LR_FROZEN_HEAD` 定義了但從未使用
- **位置**：line 58、`build_optimizer()` line 130-133
- **問題**：設計意圖是 detection head 永遠 trainable 且有獨立學習率，但 `build_optimizer` 只有一個 param group，所有 trainable 參數共用 `LR`
- **注意**：目前 `LR = LR_FROZEN_HEAD = 1e-3`，拆 param group 不會改變訓練行為。只有在設成不同值時才有實際效果。
- **注意**：不能用 `if p.requires_grad` 過濾 head params，否則 `FREEZE_LAYERS` 設太大時 head 會意外變成不訓練。應強制 head params `requires_grad=True`。
- **建議**（僅在想讓 head 用不同 LR 時才需改）：
  ```python
  head_params = list(net.model[-1].parameters())
  for p in head_params:
      p.requires_grad_(True)   # head 永遠 trainable
  head_ids = {id(p) for p in head_params}
  backbone_params = [p for p in net.parameters()
                     if p.requires_grad and id(p) not in head_ids]
  optimizer = torch.optim.AdamW([
      {"params": backbone_params, "lr": LR},
      {"params": head_params,     "lr": LR_FROZEN_HEAD},
  ], weight_decay=weight_decay)
  ```

### 🟡 `total_steps = 10_000` hardcode 導致 cosine schedule 幾乎不作用
- **位置**：line 228
- **問題**：5 分鐘內實際步數可能只有幾百步，`progress` 永遠接近 0，LR 幾乎維持在最高點不衰減
- **建議**：用預估步數替代：
  ```python
  steps_per_epoch = len(train_loader)
  estimated_epochs = max(1, TIME_BUDGET // (steps_per_epoch * avg_step_time))
  total_steps = steps_per_epoch * estimated_epochs
  ```
  或簡單地直接用 `TIME_BUDGET` 換算：`total_steps = int(TIME_BUDGET / 0.3)`（假設每步約 0.3s）

### 🟢 沒有 checkpoint 儲存
- **位置**：training loop 結束後（line 357 附近）
- **問題**：訓練完直接進 eval，若 eval 中途 crash，所有訓練結果消失
- **建議**：訓練結束後立即存檔：
  ```python
  ckpt_path = Path("checkpoint_last.pt")
  torch.save(net.state_dict(), ckpt_path)
  print(f"Checkpoint saved → {ckpt_path}")
  ```

### 🟢 `counts` 不必要移到 GPU
- **位置**：line 290
- **問題**：`counts` 只在 `counts[b].item()` 用到（CPU 操作），搬到 GPU 是多餘的
- **建議**：移除 `counts = counts.to(device, non_blocking=True)`

### 🟢 `import math` 在 for loop 內部
- **位置**：`build_model()` line 104
- **問題**：每次迭代都執行 import（Python 有快取所以不會重複載入，但寫法不乾淨）
- **建議**：移到檔案頂部的 import 區

---

## 嚴重程度彙整

| 嚴重程度 | 數量 | 問題 |
|---------|------|------|
| ✅ 已修正 | 2 | MFU 公式、eval output 解包 |
| 🔴 高 | 2 | NMS 靜默失敗、`hyp` or 陷阱 |
| 🟡 中 | 4 | `head.stride` 未初始化、scores 假設、`LR_FROZEN_HEAD` 未使用、cosine schedule 無效 |
| 🟢 低 | 3 | 無 checkpoint、`counts` 多餘搬 GPU、`import math` 位置 |
