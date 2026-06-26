#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/run/stage_common.sh"
stage_init "$@"

ARGS=("${PYTHON_BIN}" "${ROOT_DIR}/run/build_kb.py"
  --input "${INPUT_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --dataset-name "${DATASET_NAME}"
)
if [[ -n "${SAMPLE_LIMIT:-}" ]]; then
  ARGS+=(--sample-limit "${SAMPLE_LIMIT}")
fi
ARGS+=("${STAGE_REMAINING_ARGS[@]}")

stage_log "01 build_kb dataset=${DATASET_NAME} input=${INPUT_PATH}"
"${ARGS[@]}"
