---
name: deepfake-training
description: Quản lý vòng lặp huấn luyện, hàm loss, optimizer/scheduler và cấu hình Trainer cho dự án deepfake detection.
---

# Context và Thông số dự án

## Thông số trích xuất từ mã nguồn

### Training (từ `configs/train_config.yaml`)
- **batch_size:** `16`
- **num_epochs:** `50`
- **precision:** `16` (fp16 AMP)
- **gradient_clip_val:** `1.0`
- **accumulate_grad_batches:** `2` (gradient accumulation khi GPU nhỏ)
- **patience:** `7` (early stopping)
- **min_delta:** `0.001` (cải thiện tối thiểu)

### Optimizer
- **name:** `AdamW`
- **lr_backbone:** `1.0e-5` (learning rate nhỏ cho EfficientNet pretrained)
- **lr_head:** `1.0e-4` (learning rate cho Transformer head)
- **weight_decay:** `0.01`
- **betas:** `[0.9, 0.999]`
- Sử dụng differential learning rate — backbone và head có param groups riêng biệt qua `model.get_optimizer_groups()`

### Scheduler
- **name:** `CosineAnnealingLR`
- **T_max:** `50` (= num_epochs)
- **eta_min:** `1.0e-7`
- **warmup_epochs:** `3` (linear warmup trước cosine)
- Khi `warmup_epochs > 0`: dùng `SequentialLR([LinearLR, CosineAnnealingLR])` với milestone tại `warmup_epochs`

### Loss Function
- **name:** `FocalLossWithSmoothing`
- **gamma:** `2.0` (focal weight exponent)
- **alpha:** `0.25` (class balancing weight cho positive class)
- **smoothing:** `0.1` (label smoothing)
- **consistency_weight:** `0.0` (temporal consistency loss, mặc định tắt)
- Tính toán: `loss = mean(alpha_t * (1-p_t)^gamma * BCE_smooth)` + optional temporal consistency

### Checkpoint
- **save_dir:** `/root/deepfake_detector/checkpoints`
- **save_top_k:** `3` (giữ 3 checkpoint tốt nhất)
- **monitor:** `val_auc`
- **mode:** `max`
- **save_last:** `true`

### Logging
- **use_wandb:** `false`
- **wandb_project:** `deepfake-detector`
- **log_every_n_steps:** `10`

### Hardware
- **gpus:** `1`
- **strategy:** `auto` (đổi sang `ddp` nếu multi-GPU)

### Model Architecture (từ `configs/model_config.yaml`)
- **Backbone:** `EfficientNet-B4` (via `timm`, pretrained=true, feat_dim=1792)
- **Transformer Head:** d_model=512, nhead=8, num_layers=4, dim_feedforward=2048, activation=GELU, Pre-LayerNorm
- **Freeze strategy:** freeze backbone toàn bộ lúc đầu, unfreeze 3 block cuối sau epoch 5
- **num_frames:** `10`, **img_size:** `256`

### Thư viện cốt lõi
- `torch>=2.2` (AMP, GradScaler)
- `pytorch-lightning>=2.2` (DataModule integration)
- `scikit-learn>=1.3` (metrics trong validate step)
- `timm>=0.9.16` (EfficientNet backbone)

## Các file liên quan chính
| File | Vai trò |
|------|---------|
| `training/train.py` | Entrypoint script – load config, build model/optimizer/scheduler/criterion, khởi tạo Trainer, hỗ trợ resume |
| `training/trainer.py` | `Trainer` class – vòng lặp train/val thuần PyTorch, AMP, gradient clipping, early stopping, checkpoint |
| `training/loss.py` | `FocalLossWithSmoothing` – Focal Loss + Label Smoothing + optional temporal consistency |
| `models/deepfake_model.py` | `DeepfakeDetector` – model chính, cung cấp `get_optimizer_groups()` cho differential LR |
| `configs/train_config.yaml` | Toàn bộ cấu hình training |
| `configs/model_config.yaml` | Cấu hình model (backbone, transformer head, positional encoding) |

# Hướng dẫn thực thi (Instructions)

1. **Trigger:** Kích hoạt skill này khi người dùng yêu cầu viết/sửa code training, tối ưu loss function, điều chỉnh optimizer/scheduler, hoặc sửa đổi Trainer.

2. **Liên kết chặt chẽ với source code:**
   - `training/train.py` là entrypoint duy nhất — nó load cả `train_config.yaml` và `model_config.yaml` (auto-merge nếu `model` key thiếu trong train config), cũng load `aug_config.yaml` qua `load_aug_config()`.
   - `training/trainer.py` nhận model, dataloaders, optimizer, scheduler, criterion qua constructor — KHÔNG tự build. Mọi thay đổi build logic phải ở `train.py`.
   - Loss function nằm tách riêng tại `training/loss.py` với interface: `criterion(logits, targets, clip_frame_logits=None, consistency_weight=None)`.

3. **AMP và Gradient Safety:**
   - Forward pass chạy trong `autocast(enabled=self.use_amp)`.
   - Loss được tính **ngoài** autocast ở fp32 để ổn định (FIX C1).
   - Gradient flow: `scaler.scale(loss).backward()` → `scaler.unscale_(optimizer)` → kiểm tra finite gradients → `clip_grad_norm_` → `scaler.step()` → `scaler.update()`.
   - Batch có non-finite loss/gradient sẽ bị skip (có đếm và log cảnh báo).

4. **Early Stopping:**
   - Monitor `val_auc` (mode=max).
   - Chỉ lưu checkpoint khi `val_auc > best + min_delta` VÀ `val_auc > 0.5` (tốt hơn random).
   - Dừng khi `epochs_no_improve >= patience` (mặc định 7).
   - Cuối training, auto load best checkpoint vào model nếu `load_best_at_end=True`.

5. **Validation multi-clip:**
   - Val batch có thể là `[B, T, C, H, W]` hoặc `[B, N, T, C, H, W]`.
   - Loss monitor tính trên tất cả clips (flatten).
   - Metrics (AUC/F1/ACC) tính ở cấp video sau max-pooling theo clip (asymmetric: 1 clip có artifact → video fake).

6. **Backbone unfreeze strategy:**
   - `freeze_at_start=True` → đông băng toàn bộ backbone.
   - Sau `unfreeze_after_epoch=5`, mở khóa `unfreeze_last_n_blocks=3` block cuối.
   - **Lưu ý:** Logic unfreeze hiện chưa được tự động trigger trong `trainer.py` — cần implement callback hoặc hook trong training loop nếu muốn kích hoạt.

7. **Resume training:**
   - `train.py` hỗ trợ `--resume` flag để load checkpoint và tiếp tục training.
   - Resume load model_state_dict, optimizer_state_dict, scheduler_state_dict.
   - Tự động tính `remaining_epochs = num_epochs - start_epoch + 1`.

8. **Tracking và Logging:**
   - Hiện tại dùng `print()` cho mỗi epoch (train_loss, train_acc, val_loss, val_auc, val_f1, val_acc, lr).
   - W&B integration có cấu hình sẵn nhưng `use_wandb=false` — khi bật cần implement W&B logger.
   - `log_every_n_steps=10` để config cho future logger.

9. **Khi thêm tính năng mới:**
   - Luôn cập nhật cả `configs/train_config.yaml` nếu thêm hyperparameter mới.
   - Giữ backward-compatible keys (ví dụ: `gradient_clip_val` + `max_grad_norm`, `lr_backbone` trong cả `optimizer` và `training` sections).
   - Test với cả CPU và CUDA paths (AMP chỉ bật trên CUDA).
