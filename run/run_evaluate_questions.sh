#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate brj
fi

INPUT_PATH="${INPUT_PATH:-${ROOT_DIR}/outputs/pipeline/gsm8k/generated.jsonl}"
SOURCE_MAP="${SOURCE_MAP:-${ROOT_DIR}/outputs/pipeline/gsm8k/source_map.json}"
TARGET_MAP="${TARGET_MAP:-${ROOT_DIR}/outputs/pipeline/gsm8k/target_map.json}"
OUTPUT_PATH="${OUTPUT_PATH:-${ROOT_DIR}/outputs/pipeline/gsm8k/evaluated.jsonl}"

exec python "${ROOT_DIR}/run/evaluate_questions.py" \
  --input "${INPUT_PATH}" \
  --source-map "${SOURCE_MAP}" \
  --target-map "${TARGET_MAP}" \
  --output "${OUTPUT_PATH}"
