#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${ROOT_DIR}/config/pipeline.env"
EXAMPLE_CONFIG_FILE="${ROOT_DIR}/config/pipeline.example.env"

if [[ -f "${CONFIG_FILE}" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "${CONFIG_FILE}"
  set +a
elif [[ -f "${EXAMPLE_CONFIG_FILE}" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "${EXAMPLE_CONFIG_FILE}"
  set +a
fi

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate brj
fi

DATASET_NAME="${DATASET_NAME:-gsm8k}"
if [[ $# -gt 0 && "${1:-}" != --* ]]; then
  DATASET_NAME="$1"
  shift
fi

echo "[run_full_pipeline] dataset=${DATASET_NAME}"
echo "[run_full_pipeline] launching pipeline..."

exec python "${ROOT_DIR}/run/pipeline.py" \
  --input "${INPUT_PATH:-${ROOT_DIR}/data/gsm8k.jsonl}" \
  --output-dir "${OUTPUT_DIR:-${ROOT_DIR}/outputs}" \
  --dataset-name "${DATASET_NAME}" \
  "$@"
