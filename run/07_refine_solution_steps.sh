#!/usr/bin/env bash
set -euo pipefail

# Stage 07 rewrites only the validated solution-step wording. It must not
# change the question, answer, difficulty, or identifiers; those fields were
# already accepted by validation and are preserved in kb_pipeline.step_refine.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/run/stage_common.sh"
stage_init "$@"

REFINE_MODEL_NAME="${REFINE_MODEL:-${REPAIR_MODEL:-${QC_MODEL:-${GEN_MODEL:-${VLLM_MODEL:-}}}}}"

stage_require_file "${VALIDATED_OUTPUT_PATH}" "run: bash run/06_validate_generated.sh ${DATASET_NAME}"
# Refinement usually uses the same Qwen/vLLM service as validation/repair, but
# the model remains configurable for experiments.
stage_ensure_vllm "${REFINE_MODEL_NAME}" "step refinement"

if [[ -s "${REFINE_FAILED_PATH}" ]]; then
  stage_log "07 refine_solution_steps found pending failures; resume instead of skipping: ${REFINE_FAILED_PATH}"
else
  if stage_skip_if_complete "07_refine_solution_steps" "${REFINED_OUTPUT_PATH}" "${REFINE_SUMMARY_PATH}"; then
    exit 0
  fi
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
