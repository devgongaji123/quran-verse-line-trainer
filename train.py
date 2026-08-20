#!/usr/bin/env python3
"""
Training script for YOLOv26 on Quran verse line detection dataset.
Uses Ultralytics YOLO with built-in wandb integration.
"""

import os
from pathlib import Path
from ultralytics import YOLO

# Set wandb environment variables
os.environ["WANDB_PROJECT"] = "yolov26-quran-verse-line-detection"
os.environ["WANDB_LOG_MODEL"] = "true"
os.environ["WANDB_DISABLED"] = "false"

# Configuration
ROOT = Path(__file__).parent
DATA_YAML = ROOT / "data.yaml"
WEIGHTS = ROOT / "yolo26s.pt"
OUTPUT_DIR = ROOT / "runs"


def train():
    """Run YOLO training with wandb tracking."""

    # Load model
    print(f"Loading weights from: {WEIGHTS}")
    model = YOLO(str(WEIGHTS))

    # Train with wandb logging
    print(f"Training on dataset: {DATA_YAML}")
    print(f"Output directory: {OUTPUT_DIR}")

    results = model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=640,
        batch=16,
        project=str(OUTPUT_DIR),
        name="yolov26-quran-verse",
        exist_ok=True,
        patience=20,
        save=True,
        save_period=10,
        device="mps",  # macOS Apple Silicon (use "0" for CUDA GPU)
        workers=8,
        pretrained=True,
        optimizer="auto",
        verbose=True,
        seed=0,
        deterministic=True,
        plots=True,
        rect=False,
        cos_lr=False,
        close_mosaic=10,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True,
        split="val",
        save_json=False,
        save_hybrid=False,
        conf=None,
        iou=0.7,
        max_det=300,
        half=False,
        dnn=False,
        source=None,
        vid_stride=1,
        stream_buffer=False,
        visualize=False,
        augment=False,
        agnostic_nms=False,
        classes=None,
        retina_masks=False,
        embed=None,
        show=False,
        save_frames=False,
        save_txt=False,
        save_conf=False,
        save_crop=False,
        show_labels=True,
        show_conf=True,
        show_boxes=True,
        line_width=None,
        format="torchscript",
        keras=False,
        optimize=False,
        int8=False,
        dynamic=False,
        simplify=True,
        opset=None,
        workspace=None,
        nms=False,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        pose=12.0,
        kpt_shape=None,
        segment_mask=True,
        copy_paste=0.0,
        auto_augment="randaugment",
        erasing=0.4,
        crop_fraction=1.0,
        cfg=None,
        multi_scale=False,
        single_cls=False,
        nbs=64,
        mosaic=1.0,
        mixup=0.0,
        copy_paste_mode="flip",
        hydra=False,
        evolve=None,
        evolve_epochs=30,
        evolve_stop_target=None,
        evolve_parent=None,
        label_smoothing=0.0,
        syntax_check=False,
        compile=False,
        compile_force_eval=False,
        profile=False,
        freeze=None,
        multi_gpu=False,
        sync_bn=False,
        rank=-1,
        local_rank=-1,
        world_size=1,
        boxes=True,
    )

    print(f"\nTraining complete! Results saved to: {OUTPUT_DIR / 'yolov26-quran-verse'}")
    print(f"View wandb logs at: https://wandb.ai/aldisa546-stealth-company/yolov26-quran-verse-line-detection")

    return results


if __name__ == "__main__":
    train()
