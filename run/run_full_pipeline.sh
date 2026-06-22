#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/run/common_env.sh"
load_pipeline_config "${ROOT_DIR}"
activate_pipeline_env
PYTHON_BIN="$(resolve_pipeline_python)"

DATASET_NAME="${DATASET_NAME:-gsm8k}"
if [[ $# -gt 0 && "${1:-}" != --* ]]; then
  DATASET_NAME="$1"
  shift
fi

echo "[run_full_pipeline] dataset=${DATASET_NAME}"
echo "[run_full_pipeline] launching pipeline..."

exec "${PYTHON_BIN}" "${ROOT_DIR}/run/pipeline.py" \
  --input "${INPUT_PATH:-${ROOT_DIR}/data/gsm8k.jsonl}" \
  --output-dir "${OUTPUT_DIR:-${ROOT_DIR}/outputs}" \
  --dataset-name "${DATASET_NAME}" \
  "$@"
