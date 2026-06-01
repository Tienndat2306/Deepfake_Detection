---
name: deepfake-data-pipeline
description: Xử lý data loading, augmentation và chuẩn bị DataModule cho PyTorch Lightning trong dự án deepfake detection.
---

# Context và Thông số dự án

## Thông số trích xuất từ mã nguồn

### Data Pipeline (từ `configs/train_config.yaml`)
- **train_dir:** `/root/deepfake_detector/processed_faces/train`
- **val_dir:** `/root/deepfake_detector/processed_faces/val`
- **test_dir:** `/root/deepfake_detector/processed_faces/test`
- **val_split:** `0.0` (đã có val set riêng, không auto-split)
- **auto_split:** `false`
- **num_clips_eval:** `3` (số clips dùng cho val/test inference)
- **num_workers:** `8` (train), **eval_num_workers:** `12` (val/test multi-clip)
- **pin_memory:** `true`
- **prefetch_factor:** `2`
- **batch_size:** `16`

### Augmentation (từ `configs/aug_config.yaml`)
- **Preprocess:** `crop_margin=0.3`, `target_size=256`, `min_face_confidence=0.6`, `min_face_size_ratio=0.05`
- **Train augmentation:**
  - `horizontal_flip_p: 0.5`
  - `ColorJitter:` brightness=0.15, contrast=0.15, saturation=0.1, hue=0.0, p=0.8
  - `GaussianBlur:` blur_limit=[3, 7], p=0.3
  - `GaussianNoise:` var_limit=[10, 50], p=0.2
  - `JPEG Compression:` quality_lower=60, quality_upper=100, p=0.3 (đặc biệt quan trọng cho deepfake detection)
  - `RandomGrayscale:` p=0.05
  - `CoarseDropout:` max_holes=4, max_height=32, max_width=32, p=0.2
- **Val augmentation (deterministic):**
  - Resize: `image_size=256`, `resize_scale=1.14`
  - Normalize: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` (chuẩn ImageNet)

### Model Input (từ `configs/model_config.yaml`)
- **img_size:** `256`
- **num_frames:** `10`
- **channels:** `3`

### Thư viện cốt lõi (từ `requirements.txt`)
- `torch>=2.2`, `torchvision>=0.17`
- `albumentations>=1.4.0` (augmentation framework chính)
- `opencv-python-headless>=4.8.0` (I/O ảnh)
- `Pillow>=10.0`
- `pytorch-lightning>=2.2` (DataModule)
- `mediapipe>=0.10.14` (face preprocessing)
- `numpy>=1.24`, `PyYAML>=6.0`

### Cấu trúc dữ liệu
Pipeline kỳ vọng cấu trúc thư mục:
```
root_dir/
  Real/
    video_001/
      frame_0001.jpg
      ...
  Fake/
    video_002/
      frame_0001.jpg
      ...
```
- Label: `Real=0`, `Fake=1`
- Frame extensions hợp lệ: `.jpg`, `.jpeg`, `.png`

## Các file liên quan chính
| File | Vai trò |
|------|---------|
| `data/dataset.py` | `DeepfakeDataset` – Dataset class, multi-clip sampling (train: random 1 clip, val/test: N clips đầu/giữa/cuối) |
| `data/datamodule.py` | `DeepfakeDataModule` – PyTorch Lightning DataModule, tích hợp WeightedRandomSampler, stratified/group split |
| `data/augmentation.py` | `FrameAugmentation`, `ClipAugmentation`, `ClipValTransform` – Spatial-consistent augmentation qua `ReplayCompose` |
| `configs/train_config.yaml` | Cấu hình data paths, batch_size, num_workers |
| `configs/aug_config.yaml` | Cấu hình augmentation tách riêng |

# Hướng dẫn thực thi (Instructions)

1. **Trigger:** Kích hoạt skill này khi người dùng yêu cầu sửa đổi pipeline dữ liệu, augmentation, hoặc xử lý dataset (ví dụ: thêm augmentation mới, thay đổi sampling strategy, sửa DataModule).

2. **Tuân thủ logic hiện có:**
   - `DeepfakeDataset` sử dụng `clip_transform` (ưu tiên) hoặc `transform` (fallback per-frame). Khi sửa augmentation, luôn đảm bảo cả hai path hoạt động.
   - Train mode: 1 clip ngẫu nhiên liên tục, trả về `[T, C, H, W]`.
   - Val/Test mode: `num_clips_eval` clips (đầu/giữa/cuối), trả về `[N, T, C, H, W]`.
   - Video có ít frame hơn `num_frames=10` sẽ được tile theo cyclic index.

3. **Augmentation temporal consistency:**
   - Spatial augmentation (flip, rotate, resize) phải dùng `A.ReplayCompose` để giữ nhất quán giữa các frame trong cùng một clip.
   - Pixel augmentation (color jitter, blur, noise) có thể dùng `pixel_temporal_jitter > 0` để cho phép dao động nhẹ giữa các frame.
   - Val/test chỉ dùng deterministic transform: `Resize(256*1.14) → CenterCrop(256) → Normalize(ImageNet)`.

4. **Ánh xạ config:**
   - Mọi thay đổi augmentation phải được reflected trong `configs/aug_config.yaml` (không hardcode trong Python).
   - Thay đổi batch_size, num_workers, data paths → cập nhật `configs/train_config.yaml`.
   - `data/augmentation.py` đọc config qua tham số `cfg` trong `build_train_pixel_augment()`.

5. **DataModule (`data/datamodule.py`):**
   - Kế thừa `pl.LightningDataModule` (có fallback khi lightning chưa cài).
   - Hỗ trợ `use_weighted_sampler=True` bằng `WeightedRandomSampler` để cân bằng class.
   - Hỗ trợ `auto_split` + `group_by_video_id` để tránh data leakage khi tự chia train/val.
   - Khi thêm tính năng mới, giữ backward-compatible với cả `train.py` (dùng DataModule trực tiếp) và standalone usage.

6. **Kiểm tra sau khi sửa:**
   - Đảm bảo output tensor shape đúng: train `[T, C, H, W]`, val/test `[N, T, C, H, W]`.
   - Verify normalize values đúng ImageNet mean/std.
   - Kiểm tra `__len__()` trả về số video (không phải số frame).
