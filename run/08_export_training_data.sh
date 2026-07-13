#!/usr/bin/env bash
set -euo pipefail

# Stage 08 is the final SFT export. By default it prefers refined.jsonl from
# Stage 07, then falls back to validated.jsonl so older workflows still work.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/run/stage_common.sh"
stage_init "$@"

if [[ -n "${EXPORT_INPUT_PATH:-}" ]]; then
  EXPORT_INPUT="${EXPORT_INPUT_PATH}"
elif [[ -s "${REFINED_OUTPUT_PATH}" ]]; then
  # Use step-polished records when available; all non-step fields are inherited
  # from validated records by the refinement stage.
  EXPORT_INPUT="${REFINED_OUTPUT_PATH}"
else
  EXPORT_INPUT="${VALIDATED_OUTPUT_PATH}"
fi
stage_require_file "${EXPORT_INPUT}" "run: bash run/07_refine_solution_steps.sh ${DATASET_NAME} or bash run/06_validate_generated.sh ${DATASET_NAME}"

if stage_skip_if_complete "08_export_training_data" "${TRAIN_OUTPUT_PATH}" "${TRAIN_SUMMARY_PATH}"; then
  exit 0
fi

stage_log "08 export_training_data input=${EXPORT_INPUT} output=${TRAIN_OUTPUT_PATH}"
"${PYTHON_BIN}" "${ROOT_DIR}/run/build_training_data.py" \
  --input "${EXPORT_INPUT}" \
  --output "${TRAIN_OUTPUT_PATH}" \
  --summary-output "${TRAIN_SUMMARY_PATH}" \
  "${STAGE_REMAINING_ARGS[@]}"
