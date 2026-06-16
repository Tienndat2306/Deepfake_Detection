# Deepfake Video Detector

End-to-end deepfake video detection system built with PyTorch. The project covers the full workflow from video preprocessing and face extraction to model training, evaluation, and a Flask web demo for single-video inference.

## Highlights

- Detects deepfake videos from sampled face crops rather than raw video frames.
- Uses EfficientNet-B4 as a spatial feature extractor and a Transformer head for temporal aggregation.
- Includes preprocessing, training, evaluation, inference, and web demo modules.
- Supports forensic evaluation metrics such as ROC-AUC, EER, Average Precision, F1-score, confusion matrix, ROC curve, and failure analysis.
- Includes a trained checkpoint with validation AUC around `0.976999` in the current local project state.

## Project Status

This repository is suitable as a portfolio/CV project after adding external links for dataset and model artifacts. Large assets such as datasets, checkpoints, uploaded videos, and generated reports should not be committed directly to GitHub.

Recommended public repository setup:

- Commit source code, configs, notebooks, README, and small sample assets only.
- Store checkpoints in Google Drive, Hugging Face, Kaggle, or GitHub Releases.
- Document dataset source and preprocessing steps.
- Add evaluation reports under `reports/` or attach them in the README.

## Architecture

```text
Input video
  -> uniform frame sampling
  -> face detection with MediaPipe BlazeFace
  -> aligned face crop and resize
  -> sequence of face crops [T, C, H, W]
  -> EfficientNet-B4 frame feature extractor
  -> Transformer temporal head
  -> binary prediction: Real / Fake
```

Default model configuration:

| Component | Configuration |
|---|---|
| Backbone | EfficientNet-B4 via `timm` |
| Feature dimension | 1792 |
| Temporal head | Transformer encoder |
| Transformer dimension | 512 |
| Attention heads | 8 |
| Transformer layers | 4 |
| Input frames | 10 |
| Input image size | 256 x 256 |
| Output | Binary logit, sigmoid probability |

## Repository Structure

```text
deepfake_detector/
|-- app.py                         # Flask app entrypoint
|-- app/                           # Web UI, API routes, inference service
|-- configs/                       # Training, model, augmentation configs
|-- data/                          # PyTorch dataset and augmentation pipeline
|-- evaluation/                    # Evaluation script and metrics
|-- inference/                     # Frame-level inference helpers
|-- models/                        # EfficientNet + Transformer model
|-- notebooks/                     # Preprocess, EDA, train, evaluate, inference notebooks
|-- preprocess/                    # Video preprocessing and face detection
|-- training/                      # Training loop, loss, trainer
|-- clean_processed_dataset.py     # Dataset quality audit/cleanup utility
|-- resplit_dataset.py             # Train/val/test split utility
|-- check_data_leakage.sh          # Data leakage check helper
|-- requirements.txt
`-- README.md
```

## Main Features

### 1. Video Preprocessing

The preprocessing pipeline converts raw videos into face-crop frame sequences:

- reads video metadata with OpenCV;
- samples frames across the video;
- detects faces using MediaPipe BlazeFace;
- aligns and crops faces;
- stores output as image folders grouped by video ID.

## Dataset

  This project uses a processed face-crop dataset for deepfake video detection. Each original video is converted into a folder of sampled face frames, and
  Dataset download link:

  [Download Processed Dataset](https://drive.google.com/file/d/1lvF5k03NnnbD6QouUDCGSr-shstcV69o/view?usp=sharing)

  After downloading and extracting the dataset, place it in the project directory with the following structure:
  deepfake_detector/
  `-- dataset/
      |-- train/
      |   |-- Real/<video_id>/*.jpg
      |   `-- Fake/<video_id>/*.jpg
      |-- val/
      |   |-- Real/<video_id>/*.jpg
      |   `-- Fake/<video_id>/*.jpg
      `-- test/
          |-- Real/<video_id>/*.jpg
          `-- Fake/<video_id>/*.jpg

  The model does not train directly on raw videos. Instead, videos are first preprocessed into aligned face crops using MediaPipe BlazeFace. During training
  and evaluation, the dataset loader samples multiple frames from each video folder and feeds them as a temporal sequence to the model.

  Dataset notes:

  - Label Real represents authentic videos.
  - Label Fake represents manipulated/deepfake videos.
  - Each <video_id> folder contains sampled face-crop frames from one video.
  - Train/validation/test splits are organized at the video level to reduce data leakage.
  - Large dataset files are not committed to GitHub; they are provided through external storage.

### 2. Training

The training code uses:

- AdamW optimizer;
- cosine learning-rate schedule with warmup;
- mixed precision training;
- focal loss with label smoothing;
- gradient clipping;
- early stopping;
- checkpoint selection by validation AUC.

### 3. Evaluation

The evaluation pipeline supports:

- multi-clip inference;
- threshold tuning;
- ROC-AUC;
- EER;
- Average Precision;
- F1-score;
- precision and recall;
- confusion matrix;
- ROC curve plotting;
- failure report export.

### 4. Web Demo

The Flask app provides:

- video upload;
- model inference;
- Real/Fake probability;
- confidence score;
- extracted keyframes;
- video metadata;
- API endpoints for health check and session result retrieval.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/deepfake_detector.git
cd deepfake_detector
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches your CUDA version from the official PyTorch installation guide before installing the rest of the dependencies.

## Configuration

The main config files are:

- `configs/train_config.yaml`: dataset paths, optimizer, scheduler, loss, checkpoint settings;
- `configs/model_config.yaml`: EfficientNet and Transformer architecture;
- `configs/aug_config.yaml`: face preprocessing and augmentation settings.

Before training or evaluation, update the dataset paths in `configs/train_config.yaml`:

```yaml
data:
  train_dir: "dataset/train"
  val_dir: "dataset/val"
  test_dir: "dataset/test"
```

If the checkpoint is stored outside the repository, set:

```yaml
checkpoint:
  path: "path/to/model_checkpoint.pth"
```

## Usage

### Preprocess Videos

Use `preprocess/preprocess.py` to convert raw videos into face crops. The exact arguments depend on your raw dataset layout, so inspect:

```bash
python preprocess/preprocess.py --help
```

Typical goal:

```text
raw videos -> dataset/train|val|test/Real|Fake/<video_id>/*.jpg
```

### Train

```bash
python training/train.py --config configs/train_config.yaml
```

Resume from checkpoint:

```bash
python training/train.py --config configs/train_config.yaml --resume checkpoints/model.pth
```

### Evaluate

```bash
python evaluation/evaluate.py ^
  --config configs/train_config.yaml ^
  --checkpoint checkpoints/model.pth ^
  --data_dir dataset/test ^
  --output_dir reports/evaluation
```

Linux/macOS equivalent:

```bash
python evaluation/evaluate.py \
  --config configs/train_config.yaml \
  --checkpoint checkpoints/model.pth \
  --data_dir dataset/test \
  --output_dir reports/evaluation
```

### Run Web Demo

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

Useful API endpoints:

| Endpoint | Method | Description |
|---|---:|---|
| `/` or `/dashboard` | GET | Web demo UI |
| `/api/health` | GET | Runtime and model status |
| `/api/analyze` | POST | Upload and analyze a video |
| `/api/session/<session_id>` | GET | Load previous analysis result |

## Results

Current local checkpoint:

```text
dfdc_efficientnet_transformer_v1_best_epoch038_auc0_976999.pth
```

Reported validation AUC from checkpoint metadata/name:

```text
AUC ~= 0.976999
```

For a public GitHub repository, add a reproducible evaluation table after running `evaluation/evaluate.py` on the final test split:

| Metric | Value |
|---|---:|
| ROC-AUC | TBD |
| EER | TBD |
| Average Precision | TBD |
| Accuracy | TBD |
| F1-score | TBD |
| Precision | TBD |
| Recall | TBD |

Also add:

- ROC curve image;
- confusion matrix image;
- representative false positives and false negatives;
- dataset size and split statistics.

## Screenshots

Add screenshots before publishing:

```text
docs/images/dashboard.png
docs/images/result_fake.png
docs/images/result_real.png
docs/images/confusion_matrix.png
docs/images/roc_curve.png
```

Suggested README embed:

```markdown
![Dashboard](docs/images/dashboard.png)
![ROC Curve](docs/images/roc_curve.png)
```

## CV Summary

Suggested CV bullet points:

- Built an end-to-end deepfake video detection system using EfficientNet-B4 and a Transformer temporal head, covering preprocessing, training, evaluation, and Flask deployment.
- Implemented face-crop preprocessing with MediaPipe BlazeFace, multi-frame video classification, focal loss training, multi-clip evaluation, and forensic metric reporting.
- Achieved approximately `0.977` validation ROC-AUC on the project checkpoint and delivered an interactive web demo for video-level Real/Fake prediction.

## Tech Stack

- Python
- PyTorch
- Torchvision
- timm
- OpenCV
- MediaPipe
- Albumentations
- scikit-learn
- Matplotlib / Seaborn
- Flask
- Jinja2
