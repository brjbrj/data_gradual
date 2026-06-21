#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate brj
fi

DATASET_NAME="${1:-gsm8k}"
INPUT_PATH="${INPUT_PATH:-${ROOT_DIR}/data/${DATASET_NAME}.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs}"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-}"

ARGS=(python "${ROOT_DIR}/run/build_kb.py" --input "${INPUT_PATH}" --output-dir "${OUTPUT_DIR}" --dataset-name "${DATASET_NAME}")
if [[ -n "${SAMPLE_LIMIT}" ]]; then
  ARGS+=(--sample-limit "${SAMPLE_LIMIT}")
fi

exec "${ARGS[@]}"
