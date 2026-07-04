#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/run/stage_common.sh"
stage_init "$@"

REFINE_MODEL_NAME="${REFINE_MODEL:-${REPAIR_MODEL:-${QC_MODEL:-${GEN_MODEL:-${VLLM_MODEL:-}}}}}"

stage_require_file "${VALIDATED_OUTPUT_PATH}" "run: bash run/06_validate_generated.sh ${DATASET_NAME}"
stage_ensure_vllm "${REFINE_MODEL_NAME}" "step refinement"

if stage_skip_if_complete "07_refine_solution_steps" "${REFINED_OUTPUT_PATH}" "${REFINE_SUMMARY_PATH}"; then
  exit 0
fi

stage_log "07 refine_solution_steps input=${VALIDATED_OUTPUT_PATH} output=${REFINED_OUTPUT_PATH}"
"${PYTHON_BIN}" "${ROOT_DIR}/run/refine_solution_steps.py" \
  --input "${VALIDATED_OUTPUT_PATH}" \
  --output "${REFINED_OUTPUT_PATH}" \
  --failed-output "${REFINE_FAILED_PATH}" \
  --raw-output "${REFINE_RAW_OUTPUT_PATH}" \
  --summary-output "${REFINE_SUMMARY_PATH}" \
  --model "${REFINE_MODEL_NAME}" \
  "${STAGE_REMAINING_ARGS[@]}"
