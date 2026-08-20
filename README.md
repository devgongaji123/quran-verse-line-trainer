# YOLOv26 Quran Verse Line Detection

YOLO-based object detection for detecting verse lines in Quran images.

## Overview

This project trains a YOLOv26s model to detect the horizontal separator lines (maqta) between verses in Quran page images. The trained model can be used to automatically segment Quran pages into individual verses for OCR or layout analysis.

## Dataset

The dataset contains annotated Quran page images from four different mushaf editions:

| Mushaf | Description |
|--------|-------------|
| **Mina Tajwid** | Mina mushaf with tajwid color coding |
| **Mina Non Tajwid** | Mina mushaf without tajwid color coding |
| **Quba** | Quba digital mushaf |
| **Tikrar Tajwid** | Tikrar mushaf with tajwid color coding |

- **Images**: 2,266 (WebP format)
- **Split**: 1,812 train / 454 validation
- **Classes**: `maqta` (verse separator line)
- **Storage**: Cloudflare R2 via DVC

## Pretrained Model

The pretrained model weights are available on Hugging Face:

**[devgongaji/yolov26-quran-verse-line-detector](https://huggingface.co/devgongaji/yolov26-quran-verse-line-detector)**

```bash
# Download with huggingface-cli
pip install huggingface_hub
huggingface-cli download devgongaji/yolov26-quran-verse-line-detector --local-dir ./weights
```

## Quick Start

### 1. Setup Dataset

```bash
# Pull dataset from R2
R2_ACCESS_KEY_ID=<your_key> R2_SECRET_ACCESS_KEY=<your_secret> \
  ./scripts/setup_dataset_from_r2.sh

# Prepare remote training config
./scripts/prepare_remote_dataset.sh
```

### 2. Local Training

```bash
pip install ultralytics==8.4.123 opencv-python matplotlib
python train.py --epochs 100 --batch 16 --imgsz 640 --device 0
```

### 3. Docker Training (Novita GPU)

```bash
# Bootstrap Docker on Novita (run as root)
sudo ./scripts/bootstrap_novita_docker.sh

# Run training
./scripts/docker_train.sh EPOCHS=100 BATCH=16
```

## Configuration

All parameters can be set via CLI args or environment variables:

| Parameter | CLI | Env | Default |
|-----------|-----|-----|---------|
| Dataset | `--data` | `YOLO_DATA` | `data.remote.yaml` |
| Model | `--model` | `YOLO_MODEL` | `yolo26s.pt` |
| Epochs | `--epochs` | `YOLO_EPOCHS` | `100` |
| Batch size | `--batch` | `YOLO_BATCH` | `16` |
| Image size | `--imgsz` | `YOLO_IMGSZ` | `640` |
| Device | `--device` | `YOLO_DEVICE` | `0` |
| Workers | `--workers` | `YOLO_WORKERS` | `8` |
| W&B project | `--wandb-project` | `WANDB_PROJECT` | `yolov26-quran-verse-line-detection` |
| Disable W&B | `--disable-wandb` | `WANDB_DISABLED` | `false` |

## Project Structure

```
├── train.py                    # Training entrypoint
├── Dockerfile.train            # Docker image for GPU training
├── data.remote.yaml            # Dataset config (generated, gitignored)
├── scripts/
│   ├── bootstrap_novita_docker.sh   # Setup Docker on Novita cloud
│   ├── docker_train.sh              # Docker training orchestrator
│   ├── prepare_remote_dataset.sh    # Generate data.remote.yaml
│   └── setup_dataset_from_r2.sh     # Pull dataset from R2 via DVC
└── runs/                       # Training outputs (gitignored)
```

## Docker Image

Base: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`

Installed packages:
- ultralytics 8.4.123
- opencv-python 5.0.0.93
- matplotlib, pi-heif, polars, nvidia-ml-py

## License

MIT
