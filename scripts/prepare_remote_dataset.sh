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
SOURCE_PREFIX="${SOURCE_PREFIX:-/Users/syaamil/development/YOLOv26/dataset}"
EXPECTED_IMAGES="${EXPECTED_IMAGES:-2266}"
EXPECTED_LABELS="${EXPECTED_LABELS:-2266}"
EXPECTED_TRAIN="${EXPECTED_TRAIN:-1812}"
EXPECTED_VAL="${EXPECTED_VAL:-454}"

cd "$REPO_DIR"

if [[ ! -d "$DATASET_DIR" ]]; then
  echo "Missing dataset folder: $DATASET_DIR. Run scripts/setup_dataset_from_r2.sh first." >&2
  exit 1
fi
if [[ ! -f train.txt || ! -f val.txt ]]; then
  echo "train.txt and val.txt must exist in $REPO_DIR." >&2
  exit 1
fi

sed "s#${SOURCE_PREFIX}#${DATASET_DIR}#g" train.txt > train.remote.txt
sed "s#${SOURCE_PREFIX}#${DATASET_DIR}#g" val.txt > val.remote.txt

cat > data.remote.yaml <<YAML
path: ${REPO_DIR}
train: train.remote.txt
val: val.remote.txt
names: ['maqta']
YAML

find "$DATASET_DIR" -name '*.cache' -delete 2>/dev/null || true

image_count="$(find "$DATASET_DIR" -type f -name '*.webp' | wc -l | tr -d ' ')"
label_count="$(find "$DATASET_DIR" -type f -name '*.txt' | wc -l | tr -d ' ')"
train_count="$(wc -l < train.remote.txt | tr -d ' ')"
val_count="$(wc -l < val.remote.txt | tr -d ' ')"

echo "images=${image_count}/${EXPECTED_IMAGES}"
echo "labels=${label_count}/${EXPECTED_LABELS}"
echo "train=${train_count}/${EXPECTED_TRAIN}"
echo "val=${val_count}/${EXPECTED_VAL}"

[[ "$image_count" == "$EXPECTED_IMAGES" ]] || { echo "Unexpected image count." >&2; exit 2; }
[[ "$label_count" == "$EXPECTED_LABELS" ]] || { echo "Unexpected label count." >&2; exit 2; }
[[ "$train_count" == "$EXPECTED_TRAIN" ]] || { echo "Unexpected train count." >&2; exit 2; }
[[ "$val_count" == "$EXPECTED_VAL" ]] || { echo "Unexpected val count." >&2; exit 2; }

first_image="$(sed -n '1p' train.remote.txt)"
first_label="${first_image%.webp}.txt"
[[ -f "$first_image" ]] || { echo "Missing first train image: $first_image" >&2; exit 2; }
[[ -f "$first_label" ]] || { echo "Missing first train label: $first_label" >&2; exit 2; }

echo "Prepared $REPO_DIR/data.remote.yaml"
