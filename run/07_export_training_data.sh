#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/run/stage_common.sh"
stage_init "$@"

EXPORT_INPUT_PATH="${EXPORT_INPUT_PATH:-${VALIDATED_OUTPUT_PATH}}"
stage_require_file "${EXPORT_INPUT_PATH}" "run: bash run/06_validate_generated.sh ${DATASET_NAME}"

if stage_skip_if_complete "07_export_training_data" "${TRAIN_OUTPUT_PATH}" "${TRAIN_SUMMARY_PATH}"; then
  exit 0
fi

stage_log "07 export_training_data input=${EXPORT_INPUT_PATH} output=${TRAIN_OUTPUT_PATH}"
"${PYTHON_BIN}" "${ROOT_DIR}/run/build_training_data.py" \
  --input "${EXPORT_INPUT_PATH}" \
  --output "${TRAIN_OUTPUT_PATH}" \
  --summary-output "${TRAIN_SUMMARY_PATH}" \
  "${STAGE_REMAINING_ARGS[@]}"
