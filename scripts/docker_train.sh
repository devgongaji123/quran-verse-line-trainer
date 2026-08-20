#!/usr/bin/env bash
set -euo pipefail

for arg in "$@"; do
  if [[ "$arg" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
    export "$arg"
  else
    echo "Unsupported argument: $arg. Use KEY=value overrides." >&2
    exit 64
  fi
done

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_DIR="${RUN_DIR:-/workspace/quran-runs}"
MODEL_DIR="${MODEL_DIR:-/workspace/yolo26-train}"
IMAGE="${IMAGE:-quran-yolo26-train:cu124}"
CONTAINER_NAME="${CONTAINER_NAME:-quran-yolo26-train}"
MODE="${MODE:-full}"
EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-16}"
WORKERS="${WORKERS:-8}"
RUN_NAME="${RUN_NAME:-yolo26s-quran-line-labeler}"
DATA="${DATA:-/workspace/quran-verse-line-labeler/data.remote.yaml}"
MODEL="${MODEL:-/workspace/yolo26-train/yolo26s.pt}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/logs}"
DETACH="${DETACH:-1}"

if [[ "$MODE" == "smoke" ]]; then
  EPOCHS="${SMOKE_EPOCHS:-1}"
  RUN_NAME="${SMOKE_RUN_NAME:-smoke-yolo26s-quran-line-labeler}"
  DETACH="${SMOKE_DETACH:-0}"
fi

mkdir -p "$RUN_DIR" "$MODEL_DIR" "$LOG_DIR"

if [[ ! -f "${REPO_DIR}/data.remote.yaml" ]]; then
  echo "Missing data.remote.yaml. Run scripts/prepare_remote_dataset.sh first." >&2
  exit 1
fi
if [[ ! -f "$MODEL_DIR/yolo26s.pt" ]]; then
  echo "Missing model weights: $MODEL_DIR/yolo26s.pt" >&2
  exit 1
fi

docker build -f "${REPO_DIR}/Dockerfile.train" -t "$IMAGE" "$REPO_DIR"

docker run --rm --gpus all \
  -v "${REPO_DIR}:/workspace/quran-verse-line-labeler" \
  -v "${RUN_DIR}:/workspace/quran-runs" \
  -v "${MODEL_DIR}:/workspace/yolo26-train" \
  "$IMAGE" \
  python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

docker run --rm --gpus all \
  -v "${REPO_DIR}:/workspace/quran-verse-line-labeler" \
  -v "${RUN_DIR}:/workspace/quran-runs" \
  -v "${MODEL_DIR}:/workspace/yolo26-train" \
  "$IMAGE" \
  python -c "from ultralytics import YOLO; YOLO('/workspace/yolo26-train/yolo26s.pt'); print('YOLO26s OK')"

TRAIN_CMD=(
  python /workspace/quran-verse-line-labeler/train.py
  "--model=${MODEL}"
  "--data=${DATA}"
  "--epochs=${EPOCHS}"
  "--imgsz=${IMGSZ}"
  "--batch=${BATCH}"
  --device=0
  "--workers=${WORKERS}"
  --project=/workspace/quran-runs
  "--name=${RUN_NAME}"
)

if [[ "$DETACH" == "1" ]]; then
  log_file="${LOG_DIR}/${RUN_NAME}.log"
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  docker run -d --gpus all \
    --name "$CONTAINER_NAME" \
    -v "${REPO_DIR}:/workspace/quran-verse-line-labeler" \
    -v "${RUN_DIR}:/workspace/quran-runs" \
    -v "${MODEL_DIR}:/workspace/yolo26-train" \
    "$IMAGE" \
    bash -lc "cd /workspace/quran-verse-line-labeler && ${TRAIN_CMD[*]} > '${log_file}' 2>&1"
  echo "Container: $CONTAINER_NAME"
  echo "Log: $log_file"
else
  docker run --rm --gpus all \
    -v "${REPO_DIR}:/workspace/quran-verse-line-labeler" \
    -v "${RUN_DIR}:/workspace/quran-runs" \
    -v "${MODEL_DIR}:/workspace/yolo26-train" \
    "$IMAGE" \
    bash -lc "cd /workspace/quran-verse-line-labeler && ${TRAIN_CMD[*]}"
fi
