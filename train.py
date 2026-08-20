#!/usr/bin/env python3
"""Configurable YOLO training entrypoint for Quran verse line detection."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO on the Quran verse line dataset.")
    parser.add_argument("--data", default=os.getenv("YOLO_DATA", str(ROOT / "data.remote.yaml")))
    parser.add_argument("--model", default=os.getenv("YOLO_MODEL", str(ROOT / "yolo26s.pt")))
    parser.add_argument("--epochs", type=int, default=int(os.getenv("YOLO_EPOCHS", "100")))
    parser.add_argument("--batch", type=int, default=int(os.getenv("YOLO_BATCH", "16")))
    parser.add_argument("--imgsz", type=int, default=int(os.getenv("YOLO_IMGSZ", "640")))
    parser.add_argument("--device", default=os.getenv("YOLO_DEVICE", "0"))
    parser.add_argument("--workers", type=int, default=int(os.getenv("YOLO_WORKERS", "8")))
    parser.add_argument("--project", default=os.getenv("YOLO_PROJECT", str(ROOT / "runs")))
    parser.add_argument("--name", default=os.getenv("YOLO_RUN_NAME", "yolov26-quran-verse"))
    parser.add_argument("--patience", type=int, default=int(os.getenv("YOLO_PATIENCE", "20")))
    parser.add_argument("--save-period", type=int, default=int(os.getenv("YOLO_SAVE_PERIOD", "10")))
    parser.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "yolov26-quran-verse-line-detection"))
    parser.add_argument("--disable-wandb", action="store_true", default=os.getenv("WANDB_DISABLED", "false").lower() == "true")
    return parser.parse_args()


def train() -> object:
    args = parse_args()

    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_LOG_MODEL"] = os.getenv("WANDB_LOG_MODEL", "true")
    os.environ["WANDB_DISABLED"] = "true" if args.disable_wandb else "false"

    print(f"Loading weights from: {args.model}")
    print(f"Training on dataset: {args.data}")
    print(f"Output directory: {Path(args.project) / args.name}")
    print(f"Device: {args.device}")

    model = YOLO(args.model)
    return model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        exist_ok=True,
        patience=args.patience,
        save=True,
        save_period=args.save_period,
        device=args.device,
        workers=args.workers,
        pretrained=True,
        optimizer="auto",
        verbose=True,
        seed=0,
        deterministic=True,
        plots=True,
        val=True,
        split="val",
        iou=0.7,
        max_det=300,
        close_mosaic=10,
        mosaic=1.0,
        mixup=0.0,
        box=7.5,
        cls=0.5,
        dfl=1.5,
    )


if __name__ == "__main__":
    train()
