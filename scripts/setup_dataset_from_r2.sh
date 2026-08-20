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
DATASET_DIR="${DATASET_DIR:-${REPO_DIR}/dataset}"
DVC_REMOTE_NAME="${DVC_REMOTE_NAME:-r2}"
DVC_REMOTE_URL="${DVC_REMOTE_URL:-s3://quran-ocr/yolov26-quran-verse-line-detection}"
DVC_REMOTE_ENDPOINT="${DVC_REMOTE_ENDPOINT:-https://56c6b6c396404800b068220c0e451750.r2.cloudflarestorage.com}"
DVC_REMOTE_REGION="${DVC_REMOTE_REGION:-auto}"
EXPECTED_IMAGES="${EXPECTED_IMAGES:-2266}"
EXPECTED_LABELS="${EXPECTED_LABELS:-2266}"

cd "$REPO_DIR"

if ! command -v dvc >/dev/null 2>&1; then
  python3 -m pip install "dvc[s3]"
fi

if [[ ! -f dataset.dvc ]]; then
  echo "Missing dataset.dvc in $REPO_DIR." >&2
  exit 1
fi

dvc remote add --local -f "$DVC_REMOTE_NAME" "$DVC_REMOTE_URL"
dvc remote default --local "$DVC_REMOTE_NAME"
dvc remote modify --local "$DVC_REMOTE_NAME" endpointurl "$DVC_REMOTE_ENDPOINT"
dvc remote modify --local "$DVC_REMOTE_NAME" region "$DVC_REMOTE_REGION"

if [[ -n "${R2_ACCESS_KEY_ID:-}" ]]; then
  dvc remote modify --local "$DVC_REMOTE_NAME" access_key_id "$R2_ACCESS_KEY_ID"
fi
if [[ -n "${R2_SECRET_ACCESS_KEY:-}" ]]; then
  dvc remote modify --local "$DVC_REMOTE_NAME" secret_access_key "$R2_SECRET_ACCESS_KEY"
fi

if ! dvc config --local "remote.${DVC_REMOTE_NAME}.access_key_id" >/dev/null 2>&1 \
  || ! dvc config --local "remote.${DVC_REMOTE_NAME}.secret_access_key" >/dev/null 2>&1; then
  echo "Missing R2 credentials. Set R2_ACCESS_KEY_ID/R2_SECRET_ACCESS_KEY or configure DVC local remote credentials." >&2
  exit 1
fi

dvc pull dataset.dvc

if [[ ! -d "$DATASET_DIR" ]]; then
  echo "DVC pull completed, but dataset folder is missing: $DATASET_DIR" >&2
  exit 2
fi

image_count="$(find "$DATASET_DIR" -type f -name '*.webp' | wc -l | tr -d ' ')"
label_count="$(find "$DATASET_DIR" -type f -name '*.txt' | wc -l | tr -d ' ')"

echo "images=${image_count}/${EXPECTED_IMAGES}"
echo "labels=${label_count}/${EXPECTED_LABELS}"

[[ "$image_count" == "$EXPECTED_IMAGES" ]] || { echo "Unexpected image count." >&2; exit 2; }
[[ "$label_count" == "$EXPECTED_LABELS" ]] || { echo "Unexpected label count." >&2; exit 2; }

echo "Dataset ready at $DATASET_DIR"
