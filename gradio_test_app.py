#!/usr/bin/env python3
import io
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Tuple
from urllib.request import Request, urlopen

import gradio as gr
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from ultralytics import YOLO


ROOT = Path("/Users/syaamil/development/YOLOv26")
DEFAULT_IMAGE = ROOT / "mushaf_maqta_yolo26s" / "images" / "page_003.webp"
PADDLE_PYTHON = ROOT / ".venv-paddleocr" / "bin" / "python"
PADDLE_HELPER = ROOT / "run_paddleocr_det.py"
MODEL_CACHE: Dict[str, YOLO] = {}
MODEL_ERRORS: Dict[str, str] = {}
FASTER_MODEL_CACHE: Dict[str, tuple] = {}


def validate_checkpoint(checkpoint: Path) -> Tuple[bool, str]:
    try:
        YOLO(str(checkpoint))
        return True, ""
    except Exception as exc:
        return False, str(exc)


def discover_yolo_models():
    models = {}
    for checkpoint in sorted(ROOT.glob("*/best.pt")):
        model_name = checkpoint.parent.name
        ok, error = validate_checkpoint(checkpoint)
        if ok:
            models[model_name] = checkpoint
        else:
            MODEL_ERRORS[model_name] = error
    return models


def default_model_name(model_paths):
    if not model_paths:
        return None

    novita_names = [name for name in model_paths if "novita" in name.lower()]
    if novita_names:
        return sorted(novita_names)[-1]

    yolo26m_names = [name for name in model_paths if "yolo26m" in name.lower()]
    if yolo26m_names:
        return sorted(yolo26m_names)[-1]

    return sorted(model_paths)[-1]


def discover_fasterrcnn_models():
    models = {}
    for checkpoint in sorted(ROOT.glob("*/model.pth")):
        models[checkpoint.parent.name] = checkpoint
    return models


YOLO_MODEL_PATHS = discover_yolo_models()
FASTER_MODEL_PATHS = discover_fasterrcnn_models()
DEFAULT_MODEL_NAME = default_model_name(YOLO_MODEL_PATHS)
DEFAULT_FASTER_MODEL_NAME = default_model_name(FASTER_MODEL_PATHS)


def resolve_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = resolve_device()


def load_model(model_name):
    if model_name not in YOLO_MODEL_PATHS:
        if model_name in MODEL_ERRORS:
            raise gr.Error(f"Checkpoint '{model_name}' is unavailable: {MODEL_ERRORS[model_name]}")
        raise gr.Error(f"Unknown model: {model_name}")

    if model_name not in MODEL_CACHE:
        MODEL_CACHE[model_name] = YOLO(str(YOLO_MODEL_PATHS[model_name]))
    return MODEL_CACHE[model_name]


def build_fasterrcnn(num_classes):
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_fasterrcnn_model(model_name):
    if model_name not in FASTER_MODEL_PATHS:
        raise gr.Error(f"Unknown Faster R-CNN checkpoint: {model_name}")

    if model_name not in FASTER_MODEL_CACHE:
        checkpoint = torch.load(FASTER_MODEL_PATHS[model_name], map_location="cpu")
        class_names = checkpoint.get("class_names") or ["maqta"]
        model = build_fasterrcnn(num_classes=len(class_names) + 1)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(DEVICE)
        model.eval()
        FASTER_MODEL_CACHE[model_name] = (model, class_names)
    return FASTER_MODEL_CACHE[model_name]


def dataframe_from_rows(rows):
    return pd.DataFrame(rows, columns=["detector", "class_id", "confidence", "x1", "y1", "x2", "y2"])


def load_image_from_source(image, image_url):
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if image:
        return Image.open(image).convert("RGB")

    if image_url and image_url.strip():
        request = Request(image_url.strip(), headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=30) as response:
            return Image.open(io.BytesIO(response.read())).convert("RGB")

    raise gr.Error("Upload an image or provide an image URL.")


def draw_boxes(image, rows, color):
    rendered = image.copy()
    draw = ImageDraw.Draw(rendered)
    for row in rows:
        x1 = row["x1"]
        y1 = row["y1"]
        x2 = row["x2"]
        y2 = row["y2"]
        label = row["detector"]
        if row["confidence"] is not None:
            label = f"{label}:{row['confidence']:.3f}"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
        draw.text((x1 + 4, max(0, y1 - 18)), label, fill=color)
    return rendered


def predict_yolo(model_name, image, conf_threshold, iou_threshold):
    model = load_model(model_name)
    if image is None:
        return None, dataframe_from_rows([])

    results = model.predict(
        source=image,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,
    )

    result = results[0]
    rows = []

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = float(box.conf[0].item())
        class_id = int(box.cls[0].item())
        rows.append(
            {
                "detector": "yolo",
                "class_id": class_id,
                "confidence": round(confidence, 5),
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
            }
        )

    return draw_boxes(image, rows, "#d32f2f"), dataframe_from_rows(rows)


def predict_fasterrcnn(model_name, image, conf_threshold):
    model, class_names = load_fasterrcnn_model(model_name)
    if image is None:
        return None, dataframe_from_rows([])

    image_tensor = F.convert_image_dtype(F.pil_to_tensor(image), dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        outputs = model([image_tensor])[0]

    rows = []
    for box, score, label in zip(outputs["boxes"], outputs["scores"], outputs["labels"]):
        confidence = float(score.detach().cpu().item())
        if confidence < conf_threshold:
            continue
        x1, y1, x2, y2 = box.detach().cpu().tolist()
        class_id = int(label.detach().cpu().item())
        detector_name = class_names[class_id - 1] if 0 < class_id <= len(class_names) else "fasterrcnn"
        rows.append(
            {
                "detector": detector_name,
                "class_id": class_id,
                "confidence": round(confidence, 5),
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
            }
        )

    return draw_boxes(image, rows, "#2e7d32"), dataframe_from_rows(rows)


def predict_paddleocr(image):
    if image is None:
        return None, dataframe_from_rows([])
    if not PADDLE_PYTHON.exists():
        raise gr.Error(f"PaddleOCR environment not found: {PADDLE_PYTHON}")
    if not PADDLE_HELPER.exists():
        raise gr.Error(f"PaddleOCR helper not found: {PADDLE_HELPER}")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        temp_path = Path(temp_file.name)
    try:
        image.save(temp_path)
        completed = subprocess.run(
            [str(PADDLE_PYTHON), str(PADDLE_HELPER), str(temp_path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
        parsed_rows = json.loads(completed.stdout or "[]")
    except subprocess.CalledProcessError as exc:
        raise gr.Error(exc.stderr.strip() or exc.stdout.strip() or "PaddleOCR inference failed.") from exc
    finally:
        temp_path.unlink(missing_ok=True)

    rows = []
    for row in parsed_rows:
        rows.append(
            {
                "detector": "paddleocr",
                "class_id": None,
                "confidence": row.get("confidence"),
                "x1": row["x1"],
                "y1": row["y1"],
                "x2": row["x2"],
                "y2": row["y2"],
            }
        )

    return draw_boxes(image, rows, "#1565c0"), dataframe_from_rows(rows)


def load_image_preview(image_url):
    return load_image_from_source(None, image_url)


def predict(detector_type, model_name, image, image_url, conf_threshold, iou_threshold):
    image = load_image_from_source(image, image_url)

    if detector_type == "PaddleOCR Arabic det-only":
        return predict_paddleocr(image)
    if detector_type == "Faster R-CNN":
        return predict_fasterrcnn(model_name, image, conf_threshold)
    return predict_yolo(model_name, image, conf_threshold, iou_threshold)


MODEL_CHOICES = sorted(set(YOLO_MODEL_PATHS) | set(FASTER_MODEL_PATHS))
DEFAULT_UI_MODEL = DEFAULT_FASTER_MODEL_NAME or DEFAULT_MODEL_NAME


with gr.Blocks(title="Mushaf Detector Tester") as demo:
    gr.Markdown(
        """
        # Mushaf Detector Tester
        Upload a page image to test local detectors and review the predicted boxes.
        """
    )
    if MODEL_ERRORS:
        broken = "\n".join(f"- `{name}`: corrupted or unreadable checkpoint" for name in sorted(MODEL_ERRORS))
        gr.Markdown(
            f"""
            ## Unavailable Checkpoints
            These checkpoints were skipped because they could not be loaded:
            {broken}
            """
        )
    with gr.Row():
        with gr.Column(scale=1):
            detector_type = gr.Dropdown(
                choices=["Faster R-CNN", "YOLO", "PaddleOCR Arabic det-only"],
                value="Faster R-CNN" if DEFAULT_FASTER_MODEL_NAME else "YOLO",
                label="Detector",
            )
            model_name = gr.Dropdown(
                choices=MODEL_CHOICES,
                value=DEFAULT_UI_MODEL,
                label="Checkpoint",
                info="Used for YOLO and Faster R-CNN modes.",
            )
            input_image = gr.Image(
                type="pil",
                label="Input Image",
                value=str(DEFAULT_IMAGE) if DEFAULT_IMAGE.exists() else None,
            )
            image_url = gr.Textbox(
                label="Image URL",
                placeholder="https://example.com/page_001.webp",
            )
            load_url_button = gr.Button("Load URL")
            conf = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Confidence Threshold")
            iou = gr.Slider(0.05, 0.95, value=0.45, step=0.05, label="IoU Threshold")
            run_button = gr.Button("Run Detection", variant="primary")
        with gr.Column(scale=1):
            output_image = gr.Image(type="pil", label="Detections")
            output_table = gr.Dataframe(
                headers=["detector", "class_id", "confidence", "x1", "y1", "x2", "y2"],
                datatype=["str", "number", "number", "number", "number", "number", "number"],
                label="Predicted Boxes",
            )

    load_url_button.click(
        fn=load_image_preview,
        inputs=[image_url],
        outputs=[input_image],
    )

    run_button.click(
        fn=predict,
        inputs=[detector_type, model_name, input_image, image_url, conf, iou],
        outputs=[output_image, output_table],
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861)
