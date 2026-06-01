# Tài liệu bàn giao dự án Deepfake Detector

Cập nhật: 2026-05-31  
Phạm vi: quét mã nguồn trong `D:\Deep_Learning\DoAn\deepfake_detector`.

## 1. Tổng quan & Công nghệ sử dụng

### Mục đích chính

Dự án xây dựng hệ thống phát hiện video deepfake theo pipeline end-to-end:

1. Tiền xử lý video: lấy mẫu frame, phát hiện khuôn mặt, align/crop khuôn mặt và lưu thành dataset ảnh.
2. Huấn luyện mô hình phân loại Real/Fake trên chuỗi frame khuôn mặt.
3. Đánh giá checkpoint bằng các metric forensics như AUC, EER, AP, F1, confusion matrix, ROC curve và failure report.
4. Cung cấp web demo Flask để upload video, chạy inference bằng checkpoint đã huấn luyện và trả kết quả Real/Fake kèm keyframes.

Kiến trúc model chính là `EfficientNet-B4` làm backbone trích xuất đặc trưng từng frame, sau đó `TransformerHead` tổng hợp chuỗi frame theo thời gian và xuất logit nhị phân.

Dự án hiện không dùng hệ quản trị cơ sở dữ liệu. Dataset, checkpoint, kết quả upload, keyframe và report đều lưu trên filesystem.

### Công nghệ, framework, thư viện

Nguồn phiên bản chính: `requirements.txt`.

| Nhóm | Công nghệ / thư viện | Phiên bản / ràng buộc | Vai trò |
|---|---|---:|---|
| Ngôn ngữ | Python | Không pin trong repo | Toàn bộ backend, training, inference |
| Deep learning | `torch` | `>=2.2` | Tensor, training, inference, AMP |
| Deep learning | `torchvision` | `>=0.17` | Transform tiện ích, normalization |
| Model zoo | `timm` | `>=0.9.16` | Tạo EfficientNet-B4 backbone |
| Xử lý số | `numpy` | `>=1.24` | Array, sampling, metric helper |
| Dataframe | `pandas` | `>=2.0` | Tiện ích notebook / phân tích dữ liệu |
| Augmentation | `albumentations` | `>=1.4.0` | Augment ảnh/clip khi train |
| Video/ảnh | `opencv-python-headless` | `>=4.8.0` | Đọc video, crop, resize, ghi frame |
| Ảnh | `Pillow` | `>=10.0` | Đọc frame từ dataset ảnh |
| Config | `PyYAML` | `>=6.0` | Đọc `configs/*.yaml` |
| Face detection | `mediapipe` | `>=0.10.14` | BlazeFace Tasks API / TFLite |
| Evaluation | `scikit-learn` | `>=1.3` | AUC, F1, accuracy, precision, recall |
| Visualization | `matplotlib` | `>=3.7` | ROC, confusion matrix |
| Visualization | `seaborn` | `>=0.13` | Hỗ trợ plotting |
| CLI progress | `tqdm` | `>=4.66` | Progress bar |
| Report table | `tabulate` | `>=0.9` | In bảng metric |
| Optional data module | `pytorch-lightning` | `>=2.2` | Chỉ dùng trong `data/datamodule.py` |
| Notebook | `ipython`, `ipykernel`, `jupyterlab` | `>=8.0`, `>=6.29`, `>=4.2` | EDA / thử nghiệm |
| Web | `Flask`, `werkzeug` | Không có trong `requirements.txt` | Web app đang import trực tiếp trong `app/`; cần bổ sung vào requirements |

Các file cấu hình chính:

| File | Nội dung |
|---|---|
| `configs/train_config.yaml` | Đường dẫn data, batch size, epoch, optimizer AdamW, cosine scheduler, focal loss, checkpoint |
| `configs/model_config.yaml` | Cấu hình EfficientNet-B4, Transformer, positional encoding, input frame size |
| `configs/aug_config.yaml` | Cấu hình crop mặt, confidence detector và augmentation |

## 2. Cấu trúc thư mục

```text
deepfake_detector/
|-- app.py
|-- app/
|   |-- __init__.py
|   |-- routes.py
|   |-- inference.py
|   |-- static/
|   |   |-- css/style.css
|   |   |-- js/main.js
|   |   |-- js/player.js
|   |   `-- uploads/
|   `-- templates/
|       |-- base.html
|       |-- index.html
|       |-- components/
|       `-- pages/
|-- checkpoints/
|-- configs/
|   |-- train_config.yaml
|   |-- model_config.yaml
|   `-- aug_config.yaml
|-- data/
|   |-- dataset.py
|   |-- augmentation.py
|   `-- datamodule.py
|-- dataset/
|-- evaluation/
|   |-- evaluate.py
|   `-- metrics.py
|-- inference/
|   `-- predict.py
|-- models/
|   |-- deepfake_model.py
|   |-- efficientnet.py
|   |-- transformer_head.py
|   `-- pos_encoding.py
|-- notebooks/
|-- preprocess/
|   |-- preprocess.py
|   |-- face_detector.py
|   |-- video_utils.py
|   `-- models/
|-- training/
|   |-- train.py
|   |-- trainer.py
|   `-- loss.py
|-- clean_processed_dataset.py
|-- resplit_dataset.py
|-- check_data_leakage.sh
|-- requirements.txt
|-- RTK.md
`-- HANDOVER.md
```

### Chức năng từng thư mục lớn

| Thư mục | Chức năng |
|---|---|
| `app/` | Flask web demo: upload video, gọi inference service, render UI và trả JSON API |
| `configs/` | Cấu hình model, training, augmentation và preprocess |
| `data/` | Dataset PyTorch, augmentation, DataLoader/DataModule |
| `evaluation/` | Script đánh giá checkpoint và tiện ích metric/report |
| `inference/` | Helper inference-time frame preprocessing |
| `models/` | Định nghĩa model EfficientNet + Transformer |
| `preprocess/` | Tiền xử lý video thành face crops; detector MediaPipe; utility đọc/lấy mẫu video |
| `training/` | Entrypoint train, trainer loop, loss function |
| `checkpoints/` | Checkpoint `.pth`; hiện có checkpoint best AUC khoảng `0.976999` |
| `dataset/` | Dữ liệu dự án; repo hiện không thể hiện đầy đủ nội dung dataset qua code |
| `notebooks/` | Notebook theo các bước preprocess, EDA, train, evaluate, inference |
| `app/static/uploads/` | Dữ liệu runtime do web app tạo: video upload, keyframe, `result.json`; nên cân nhắc đưa vào `.gitignore` |

## 3. Bản đồ chức năng của file

### Entrypoint và web app

| File | Nhiệm vụ chính | Hàm / class quan trọng |
|---|---|---|
| `app.py` | Entrypoint Flask local; tạo app và chạy `127.0.0.1:5000` debug mode | `create_app()` từ package `app` |
| `app/__init__.py` | Flask app factory; cấu hình upload size/folder; đăng ký blueprint | `create_app()` |
| `app/routes.py` | Định nghĩa route web và API; lazy-load inference service; lưu video upload và result JSON | `_get_service()`, `_default_context()`, `dashboard()`, `analyze()`, `health()`, `session_result()` |
| `app/inference.py` | Inference service cho Flask: load config, checkpoint, model, face detector; phân tích một video | `AppPaths`, `DeepfakeInferenceService`, `_load_model()`, `_load_face_detector()`, `analyze_video()` |
| `app/static/js/main.js` | Logic frontend: chọn file, preview video, POST `/api/analyze`, cập nhật score/keyframes/modal | `updateResult()`, `setLoading()`, `openFrameModal()` |
| `app/static/js/player.js` | File hiện rỗng | Chưa có |
| `app/templates/base.html` | Layout HTML nền | Jinja blocks |
| `app/templates/index.html` | Màn hình chính upload, preview, result, metadata, keyframes | DOM ids được `main.js` sử dụng |
| `app/templates/components/*.html` | Component UI cho layout dashboard nâng cao | Header, sidebar, video player, analysis panel, score, keyframes |
| `app/templates/pages/*.html` | Các page template dashboard/forensics/deepscan/network | Chủ yếu phục vụ UI/navigation |

### Model

| File | Nhiệm vụ chính | Hàm / class quan trọng |
|---|---|---|
| `models/deepfake_model.py` | Model tổng hợp: nhận clip `[B,T,C,H,W]`, flatten qua backbone, reshape thành sequence và đưa vào Transformer head | `DeepfakeDetector`, `forward()`, `get_optimizer_groups()`, `save_checkpoint()`, `load_checkpoint()` |
| `models/efficientnet.py` | Wrapper `timm.create_model("efficientnet_b4")`; bỏ classification head, lấy feature vector 1792 chiều | `EfficientNetExtractor`, `freeze_backbone()`, `unfreeze_last_n_blocks()`, `forward()` |
| `models/transformer_head.py` | Transformer encoder theo thời gian với CLS token, Pre-LN, optional relative position bias, stochastic depth | `DropPath`, `TemporalTransformerEncoderLayer`, `TransformerHead`, `forward()` |
| `models/pos_encoding.py` | Positional encoding học được, khởi tạo từ sinusoidal; relative bias 1D cho attention | `TemporalPositionalEncoding`, `RelativePositionBias1D` |

### Data và augmentation

| File | Nhiệm vụ chính | Hàm / class quan trọng |
|---|---|---|
| `data/dataset.py` | PyTorch Dataset cho cấu trúc `Real/Fake/video_id/*.jpg`; chọn frame theo train/val/test; hỗ trợ multi-clip eval | `DeepfakeDataset`, `_build_index()`, `_select_train_clip_paths()`, `_select_clip_paths()`, `__getitem__()`, `get_labels()`, `get_video_ids()` |
| `data/augmentation.py` | Pipeline augmentation frame/clip bằng Albumentations; giữ spatial augmentation nhất quán theo clip | `FrameAugmentation`, `ClipAugmentation`, `ClipValTransform`, `get_train_clip_transform()`, `get_val_clip_transform()`, `apply_train_transform_consistent()` |
| `data/datamodule.py` | DataModule tương thích PyTorch Lightning; split stratified/group split; WeightedRandomSampler | `DeepfakeDataModule`, `_stratified_split_indices()`, `_stratified_group_split_indices()`, `setup()`, `train_dataloader()`, `val_dataloader()` |

### Training

| File | Nhiệm vụ chính | Hàm / class quan trọng |
|---|---|---|
| `training/train.py` | Entrypoint training CLI: đọc YAML, build dataset/dataloader/model/optimizer/scheduler/loss/trainer, resume checkpoint | `parse_args()`, `load_yaml_config()`, `load_aug_config()`, `set_seed()`, `infer_feat_dim()`, `build_checkpoint_path()`, `maybe_resume()`, `main()` |
| `training/trainer.py` | Training loop PyTorch thuần: AMP, gradient accumulation, grad clipping, validation multi-clip, early stopping, save top-k checkpoint | `Trainer`, `train_one_epoch()`, `validate()`, `_maybe_unfreeze_backbone()`, `_save_best_checkpoint()`, `fit()` |
| `training/loss.py` | Loss nhị phân kết hợp Focal Loss + Label Smoothing; có hook temporal consistency optional | `FocalLossWithSmoothing`, `_temporal_consistency_loss()`, `forward()` |

### Evaluation và inference helper

| File | Nhiệm vụ chính | Hàm / class quan trọng |
|---|---|---|
| `evaluation/evaluate.py` | CLI đánh giá checkpoint: resolve data/checkpoint, chạy TTA, tune threshold trên val nếu có, xuất JSON/ROC/confusion/failure report | `resolve_checkpoint_path()`, `build_test_loader()`, `run_inference_once()`, `build_tta_transforms()`, `run_tta_inference()`, `build_failure_analysis()`, `main()` |
| `evaluation/metrics.py` | Tính metric forensics và plot | `compute_metrics()`, `find_optimal_threshold()`, `plot_roc_curve()`, `plot_confusion_matrix()` |
| `inference/predict.py` | Helper preprocess một frame inference: detect/crop mặt từ frame BGR chưa resize | `preprocess_frame_for_inference()` |

### Preprocess video

| File | Nhiệm vụ chính | Hàm / class quan trọng |
|---|---|---|
| `preprocess/preprocess.py` | CLI tiền xử lý raw video thành thư mục face crops; multiprocessing; resume an toàn bằng `.done.json`; retry detection quanh frame miss | `parse_args()`, `load_preprocess_settings()`, `_process_single_video()`, `_detect_and_crop()`, `_crop_with_retry()`, `run_preprocess_pipeline()`, `main()` |
| `preprocess/face_detector.py` | Wrapper MediaPipe BlazeFace Tasks API; tự tải model nếu thiếu; multi-scale detect, fallback confidence, align/crop bằng mắt và bbox | `FaceDetector`, `download_model_if_needed()`, `detect()`, `align_and_crop()`, `crop_from_bbox()`, `visualize_detection()` |
| `preprocess/video_utils.py` | Tiện ích video OpenCV: list video, xác thực frame count, lấy mẫu frame, đọc frame sparse, kiểm tra corrupt video | `list_video_files()`, `get_video_info()`, `sample_frame_indices()`, `read_frames_by_indices()`, `is_video_corrupted()` |
| `preprocess/models/*.tflite` | Model BlazeFace short/full range cho MediaPipe | Asset runtime |

### Script tiện ích và cấu hình

| File | Nhiệm vụ chính | Hàm / class quan trọng |
|---|---|---|
| `clean_processed_dataset.py` | Audit dataset đã preprocess; dry-run hoặc xóa folder/frame lỗi: empty, too few frames, black, uniform, blurry, optional face confidence; hỗ trợ checkpoint resume | `collect_video_folders()`, `parallel_scan()`, `parallel_delete()`, `parallel_delete_files()`, `main()` |
| `resplit_dataset.py` | Gom source train/val/test và chia lại 70/15/15 theo class; tránh trùng `video_id`; có dry-run và swap root | `VideoEntry`, `collect_entries()`, `stratified_split()`, `move_entries()`, `swap_roots()`, `main()` |
| `check_data_leakage.sh` | Script shell kiểm tra leakage dữ liệu | Shell script |
| `configs/train_config.yaml` | Cấu hình training chính | `experiment`, `data`, `training`, `optimizer`, `scheduler`, `loss`, `checkpoint` |
| `configs/model_config.yaml` | Cấu hình kiến trúc model và variants | `model`, `model_variants.lightweight`, `model_variants.full` |
| `configs/aug_config.yaml` | Cấu hình preprocess và augmentation | `preprocess`, `augmentation`, `val_augmentation` |
| `requirements.txt` | Dependency Python | Lưu ý thiếu `Flask` |
| `RTK.md` / `AGENTS.md` | Chỉ dẫn local yêu cầu dùng `rtk`; hiện `rtk` không có trong PATH khi kiểm tra | Không ảnh hưởng runtime app |

## 4. Luồng hoạt động chính

### 4.1 Luồng web app khi khởi chạy

```text
app.py
  -> app.create_app()
      -> Flask(...)
      -> config MAX_CONTENT_LENGTH, UPLOAD_FOLDER
      -> import app.routes.bp
      -> register_blueprint(bp)
  -> app.run(host="127.0.0.1", port=5000, debug=True)
```

Khi người dùng mở UI:

```text
GET /
  -> app/routes.py:dashboard()
  -> render_template("index.html", context mặc định)
  -> browser load app/static/js/main.js + CSS
```

Khi người dùng upload video và bấm Analyze:

```text
Browser main.js
  -> POST /api/analyze với FormData(video)
  -> routes.analyze()
      -> validate file extension
      -> lưu video vào app/static/uploads/<session>_<filename>
      -> tạo session_dir app/static/uploads/<session_id>/
      -> _get_service()
          -> lazy init DeepfakeInferenceService(root_dir)
      -> service.analyze_video(video_path, output_dir)
          -> load train_config.yaml + model_config.yaml fallback nếu cần
          -> resolve checkpoint trong checkpoint.save_dir hoặc checkpoint.path
          -> build DeepfakeDetector
          -> model.load_checkpoint(...)
          -> load FaceDetector BlazeFace
          -> get_video_info(video)
          -> sample_frame_indices(total_frames, num_frames*3, uniform)
          -> read_frames_by_indices(...)
          -> detect face từng frame, align/crop, lưu thumbnail keyframe
          -> nếu thiếu frame thì lặp lại crop đã có cho đủ num_frames
          -> get_val_clip_transform(img_size)
          -> model forward: [1,T,C,H,W] -> logit -> sigmoid
          -> verdict Fake nếu probability >= 0.5, ngược lại Real
          -> ghi result.json
      -> routes.analyze() thêm video_url, ghi result.json lần nữa
      -> trả JSON cho browser
  -> main.js:updateResult()
      -> cập nhật Fake/Real probability, confidence, metadata, keyframes
```

### 4.2 Luồng training

```text
python -m training.train --config configs/train_config.yaml
  -> parse_args()
  -> load_yaml_config()
      -> merge model_config.yaml nếu train_config thiếu key model
  -> load_aug_config()
  -> set_seed()
  -> build DeepfakeDataset train/val
      -> đọc dataset dạng Real/Fake/video_id/*.jpg
      -> train: chọn 1 clip/video
      -> val: chọn num_clips_eval clip/video
  -> build DataLoader + WeightedRandomSampler
  -> build DeepfakeDetector
      -> EfficientNetExtractor
      -> TransformerHead
  -> build AdamW param groups backbone/head
  -> build scheduler CosineAnnealingLR hoặc warmup + cosine
  -> build FocalLossWithSmoothing
  -> maybe_resume()
  -> Trainer.fit()
      -> train_one_epoch()
          -> forward [B,T,C,H,W]
          -> AMP nếu CUDA
          -> loss fp32
          -> grad accumulation, clipping, optimizer step
      -> validate()
          -> hỗ trợ [B,N,T,C,H,W]
          -> forward từng clip
          -> aggregate probability theo mean/max ở cấp video
          -> AUC/F1/ACC
      -> scheduler.step()
      -> save top-k checkpoint nếu val_auc_mean cải thiện và > 0.5
      -> early stopping theo patience
```

### 4.3 Luồng preprocess dataset

```text
python -m preprocess.preprocess --input_dir ... --output_dir ... --label Real|Fake --config configs/aug_config.yaml
  -> load_preprocess_settings()
  -> list_video_files(input_dir)
  -> bỏ qua video đã có đủ frame + .done.json khớp metadata
  -> ProcessPoolExecutor xử lý từng video
      -> get_video_info()
      -> sample_frame_indices(total_frames, samples_per_video * oversample_factor, uniform)
      -> read frames sparse
      -> FaceDetector.detect()
      -> align_and_crop()
      -> retry quanh frame miss trong cửa sổ +/- retry_window
      -> ghi temp folder
      -> nếu đủ samples_per_video thì replace atomic sang output_dir/Real|Fake/video_stem/
      -> ghi .done.json
```

### 4.4 Luồng evaluation

```text
python -m evaluation.evaluate --config configs/train_config.yaml --output_dir reports
  -> resolve_checkpoint_path()
  -> resolve_data_dir()
  -> build DeepfakeDetector + load checkpoint
  -> nếu có val_dir:
      -> run TTA trên val
      -> find_optimal_threshold theo F1
  -> run TTA trên test
      -> các transform deterministic: center, hflip, crop 0.95, crop 1.05
      -> average score qua TTA
  -> compute_metrics threshold=0.5 và threshold tối ưu
  -> lưu results.json, roc_curve.png, confusion_matrix.png
  -> lưu failure_report.json/csv nếu bật
```

## 5. Đánh giá hiện trạng & Gợi ý bước tiếp theo

### Tính năng đã tương đối hoàn thiện

| Nhóm | Hiện trạng |
|---|---|
| Preprocess video | Có pipeline multiprocessing, retry face detect, resume bằng metadata, ghi output atomic |
| Face detection/crop | Có MediaPipe BlazeFace, fallback confidence, multi-scale, align/crop, guard crop rỗng |
| Dataset/DataLoader | Có dataset Real/Fake/video_id, train/val/test mode, multi-clip eval, weighted sampler |
| Model | Kiến trúc rõ: EfficientNet-B4 + Transformer + CLS token + checkpoint save/load |
| Training | Có AMP, gradient clipping, accumulation, scheduler, early stopping, save top-k, resume một phần |
| Evaluation | Có TTA, tune threshold trên val, metric/report/plot/failure analysis |
| Web demo | Có Flask UI upload video, inference lazy-load model, trả score/keyframes/metadata |

### Phần còn dang dở hoặc cần làm cứng

| Vấn đề | Quan sát |
|---|---|
| `requirements.txt` thiếu dependency web | Code import `flask` và `werkzeug` nhưng requirements không liệt kê `Flask`. Lập trình viên mới có thể cài thiếu và app không chạy |
| Cấu hình path đang hardcode môi trường Linux | `configs/train_config.yaml` dùng `/root/ai_env/deepfake_detector/...`, không khớp workspace Windows hiện tại |
| `rtk` được yêu cầu nhưng không có trong PATH | `RTK.md` yêu cầu prefix `rtk`, nhưng khi kiểm tra thì `rtk` không được nhận diện |
| Web app chưa có queue/job async | `/api/analyze` chạy inference đồng bộ trong request; video dài hoặc CPU mode dễ timeout/block worker |
| Upload/runtime artifact nằm trong repo | `app/static/uploads/` chứa nhiều output runtime; nên đưa vào `.gitignore` và tách storage runtime |
| UI có nhiều template component/page nhưng route đang render chung `index.html` | Các file trong `templates/components` và `templates/pages` có vẻ phục vụ thiết kế mở rộng, chưa phải routing hoàn chỉnh |
| `app/static/js/player.js` rỗng | Có thể là placeholder |
| Preprocess file có dấu hiệu lỗi encoding comment | Một số comment trong `preprocess/*.py` hiển thị mojibake, không ảnh hưởng logic nhưng gây khó bảo trì |
| Evaluation dùng transform PIL trong `evaluate.py` khác pipeline clip transform mới | Training val dùng clip transform BGR/Albumentations; evaluate CLI dùng `DeterministicTTATransform` trên PIL. Cần xác nhận consistency nếu metric lệch |
| Checkpoint save/load chưa lưu đầy đủ optimizer khi dùng `DeepfakeDetector.save_checkpoint()` | `Trainer._save_best_checkpoint()` ưu tiên `model.save_checkpoint()`, payload chỉ có epoch, val_auc, model_state_dict; resume optimizer/scheduler chỉ hoạt động nếu checkpoint được lưu bằng fallback |

### 3-5 đầu việc kỹ thuật tiếp theo

1. Chuẩn hóa dependency và môi trường chạy.
   - Bổ sung `Flask` vào `requirements.txt`.
   - Ghi rõ Python version khuyến nghị.
   - Thêm `.env.example` hoặc config local cho data/checkpoint paths.
   - Xử lý hoặc bỏ yêu cầu `rtk` nếu tool không được cài trong môi trường nhóm.

2. Tách runtime artifacts khỏi repository.
   - Thêm `.gitignore` cho `__pycache__/`, `app/static/uploads/`, report output, checkpoint lớn nếu không muốn version.
   - Cấu hình `UPLOAD_FOLDER` qua env var.
   - Thêm cleanup policy cho upload/session cũ.

3. Làm cứng web inference cho production/demo ổn định hơn.
   - Chạy analyze qua background job hoặc queue thay vì request đồng bộ.
   - Thêm timeout, giới hạn duration/codec, validate MIME thực tế.
   - Chuẩn hóa error message tiếng Việt/tiếng Anh.
   - Thêm endpoint progress nếu inference lâu.

4. Đồng bộ pipeline train/evaluate/inference.
   - Xác nhận transform và frame sampling giữa `training/train.py`, `evaluation/evaluate.py`, `app/inference.py`.
   - Viết test nhỏ đảm bảo input tensor shape/normalize giống nhau.
   - Cân nhắc gom logic build model/config/checkpoint vào module dùng chung để tránh duplicate `infer_feat_dim`, `resolve_checkpoint_path`.

5. Bổ sung test và tài liệu vận hành.
   - Unit test cho `sample_frame_indices`, `DeepfakeDataset`, `FocalLossWithSmoothing`, `FaceDetector` wrapper bằng mock.
   - Smoke test app: `/api/health`, upload video nhỏ giả lập hoặc monkeypatch service.
   - README chạy nhanh: preprocess, train, evaluate, run web app, cấu trúc dataset chuẩn.

