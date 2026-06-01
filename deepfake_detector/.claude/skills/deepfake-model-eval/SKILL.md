---
name: deepfake-model-eval
description: Xây dựng kiến trúc model, đánh giá hiệu năng và tính toán metrics cho dự án deepfake detection.
---

# Context và Thông số dự án

## Thông số trích xuất từ mã nguồn

### Kiến trúc Model chính (từ `configs/model_config.yaml` + source code)
- **Tên model:** `DeepfakeDetector_v1`
- **Pipeline:** `EfficientNetExtractor` (backbone) → `TransformerHead` (classification head)
- **Input shape:** `[B, T=10, C=3, H=256, W=256]`
- **Output shape:** `[B]` (raw logits, chưa sigmoid)

### Backbone — `EfficientNet-B4` (`models/efficientnet.py`)
- **Thư viện:** `timm` (`timm.create_model("efficientnet_b4", pretrained=True, num_classes=0)`)
- **Feature dimension:** `1792` (output sau global average pooling)
- **Dropout:** `0.3`
- **Optional:** Global Context Pooling (concat avg+max pool → Linear projection)
- **Freeze/Unfreeze:**
  - `freeze_at_start: true` — đông băng toàn bộ backbone
  - `unfreeze_last_n_blocks: 3` — mở 3 block cuối sau warmup
  - `unfreeze_after_epoch: 5`

### Transformer Head (`models/transformer_head.py`)
- **d_model:** `512`
- **nhead:** `8`
- **num_layers:** `4`
- **dim_feedforward:** `2048` (= d_model × 4)
- **dropout:** `0.3`
- **activation:** `GELU`
- **norm_first:** `true` (Pre-LayerNorm, ổn định hơn Post-LN)
- **Stochastic depth:** `DropPath` với drop rate tăng dần từ 0 → `drop_path_rate` qua các layer
- **CLS token:** learnable, init std=0.02, dùng để tổng hợp thông tin toàn clip
- **Classifier:** `Linear(d_model=512, 1)` → raw logit `[B]`

### Positional Encoding (`models/pos_encoding.py`)
- **type:** `learnable_sinusoidal` (khởi tạo sinusoidal, fine-tune được)
- **max_len:** `32` (lớn hơn num_frames=10 để generalize)
- **dropout:** `0.1`
- **Optional:** `RelativePositionBias1D` (relative position bias cho attention)

### Model Variants (preset trong config)
| Variant | Backbone | feat_dim | d_model | num_layers | dim_feedforward |
|---------|----------|----------|---------|------------|-----------------|
| **full** (mặc định) | EfficientNet-B4 | 1792 | 512 | 4 | 2048 |
| **lightweight** | EfficientNet-B0 | 1280 | 256 | 2 | 1024 |

### Evaluation Metrics (từ `evaluation/metrics.py`)
Hàm `compute_metrics()` trả về dict gồm:
| Metric | Ý nghĩa |
|--------|---------|
| `auc_roc` | Area Under ROC Curve — metric chính để đánh giá phân loại |
| `eer` | Equal Error Rate — điểm FAR=FRR, quan trọng trong forensics |
| `eer_threshold` | Threshold tại điểm EER |
| `ap` | Average Precision (area under PR curve) |
| `accuracy` | Accuracy tại threshold đã chọn |
| `precision` | Precision tại threshold |
| `recall` | Recall tại threshold |
| `f1` | F1-Score tại threshold |

### Evaluation Pipeline (từ `evaluation/evaluate.py`)
- **TTA (Test-Time Augmentation):** 4 transforms forensic-safe — original, hflip, center_crop_0.95, center_crop_1.05
- **Multi-clip inference:** `num_clips_eval=3`, max-pooling probability theo clip
- **Threshold tuning:** Tune optimal threshold trên val set theo F1, áp dụng cho test
- **Failure analysis:** Export top-K mẫu dự đoán sai nhất ra JSON + CSV
- **Visualization:** ROC curve + Confusion matrix lưu ra PNG

### Thư viện cốt lõi
- `torch>=2.2`, `timm>=0.9.16` (backbone)
- `scikit-learn>=1.3` (metrics: roc_auc_score, f1_score, accuracy_score, etc.)
- `matplotlib>=3.7`, `seaborn>=0.13` (visualization)
- `tabulate>=0.9` (formatted metric output)

## Các file liên quan chính
| File | Vai trò |
|------|---------|
| `models/deepfake_model.py` | `DeepfakeDetector` — model chính, kết hợp backbone + head, checkpoint save/load |
| `models/efficientnet.py` | `EfficientNetExtractor` — EfficientNet-B4 backbone wrapper, freeze/unfreeze API |
| `models/transformer_head.py` | `TransformerHead` — Transformer encoder với CLS token, Pre-LN, DropPath |
| `models/pos_encoding.py` | `TemporalPositionalEncoding`, `RelativePositionBias1D` |
| `evaluation/metrics.py` | `compute_metrics()`, `find_optimal_threshold()`, `plot_roc_curve()`, `plot_confusion_matrix()` |
| `evaluation/evaluate.py` | Evaluation entrypoint — TTA, multi-clip inference, threshold tuning, failure analysis |
| `configs/model_config.yaml` | Cấu hình kiến trúc model và preset variants |

# Hướng dẫn thực thi (Instructions)

1. **Trigger:** Kích hoạt skill này khi người dùng yêu cầu:
   - Chỉnh sửa kiến trúc mạng (thay backbone, sửa transformer head, thêm module)
   - Tính toán hoặc thêm chỉ số đánh giá mới
   - Sửa đổi evaluation pipeline (TTA, threshold tuning, failure analysis)
   - Switch model variant (full ↔ lightweight)

2. **Backbone hiện tại: EfficientNet-B4 (KHÔNG phải ConvNeXt V2 hay model khác)**
   - Tên timm: `"efficientnet_b4"`, pretrained, `num_classes=0`, `global_pool="avg"`
   - Output: vector 1792-dim cho mỗi frame
   - Khi thay đổi backbone, **phải** cập nhật `feat_dim` trong model config tương ứng (ví dụ: EfficientNet-B0 → 1280)
   - API freeze/unfreeze: `backbone.freeze_backbone()`, `backbone.unfreeze_last_n_blocks(n)`

3. **Transformer Head flow:**
   ```
   Input [B, T, 1792] → Linear Projection [B, T, 512] → Prepend CLS [B, T+1, 512]
   → Positional Encoding → 4× TemporalTransformerEncoderLayer (Pre-LN + DropPath)
   → Final LayerNorm → CLS token [B, 512] → Dropout → Linear(512, 1) → logit [B]
   ```
   - Khi sửa TransformerHead, đảm bảo `dim_feedforward = d_model * 4` trừ khi có lý do cụ thể.
   - CLS token khởi tạo bằng `nn.init.normal_(std=0.02)`.

4. **Chuẩn hóa đầu ra metrics:**
   - Hàm `compute_metrics()` nhận `(y_true, y_scores, threshold)` và trả về dict đầy đủ 9 metrics.
   - `find_optimal_threshold()` scan threshold từ 0.1 đến 0.9 (bước 0.01) theo F1.
   - EER được tính bằng nội suy tuyến tính trên đường ROC (FPR ∩ FNR).
   - Khi thêm metric mới, thêm vào dict output của `compute_metrics()` và cập nhật `print_metrics_table()` trong `evaluate.py`.

5. **Evaluation pipeline:**
   - `evaluate.py` auto-merge `model_config.yaml` nếu `model` key thiếu trong train config.
   - Checkpoint resolution: `--checkpoint` → `checkpoint.path` → best checkpoint trong `checkpoint.save_dir` (theo val_auc).
   - TTA chỉ dùng biến đổi hình học nhẹ (forensic-safe), **KHÔNG** dùng color jitter vì có thể phá huỷ deepfake artifacts.
   - Multi-clip: max-pooling probability `probs.max(dim=1)` — asymmetric assumption (1 clip fake → video fake).

6. **Khi thêm module/layer mới vào model:**
   - Đảm bảo tương thích với `model.get_optimizer_groups()` (tách backbone vs head params).
   - Checkpoint save/load qua `model.save_checkpoint()` / `model.load_checkpoint()` — dùng `strict=False` để load.
   - Cập nhật `configs/model_config.yaml` nếu thêm hyperparameter mới.
   - Cập nhật cả variant `lightweight` và `full` nếu applicable.

7. **Visualization output:**
   - ROC curve: lưu `roc_curve.png` (dpi=200, 7×6 inches).
   - Confusion matrix: lưu `confusion_matrix.png` (dpi=200, 6×5 inches, Blues colormap).
   - Failure report: `failure_report.json` + `failure_report.csv` với fields: `video_path, y_true, y_score, y_pred, error_margin, error_type (FP/FN)`.
