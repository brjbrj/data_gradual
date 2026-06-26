#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/run/stage_common.sh"
stage_init "$@"

QC_MODEL_NAME="${QC_MODEL:-${QUALITY_MODEL:-${STEP_MODEL:-${VLLM_MODEL:-}}}}"

stage_require_file "${GENERATED_OUTPUT_PATH}" "run: bash run/05_generate_questions.sh ${DATASET_NAME}"
stage_require_file "${PLAN_PATH}" "run: bash run/04_build_synthesis_plan.sh ${DATASET_NAME}"
stage_require_file "${MASTERY_PATH}" "run: bash run/03_score_seed.sh ${DATASET_NAME}"
stage_ensure_vllm "${QC_MODEL_NAME}" "validation"

stage_log "06 validate_generated output=${VALIDATED_OUTPUT_PATH}"
"${PYTHON_BIN}" "${ROOT_DIR}/run/validate_generated.py" \
  --generated "${GENERATED_OUTPUT_PATH}" \
  --plan "${PLAN_PATH}" \
  --mastery "${MASTERY_PATH}" \
  --output "${VALIDATED_OUTPUT_PATH}" \
  --reports-output "${VALIDATION_REPORTS_PATH}" \
  --failed-output "${VALIDATION_FAILED_PATH}" \
  --repair-history-output "${REPAIR_HISTORY_PATH}" \
  --model "${QC_MODEL_NAME}" \
  "${STAGE_REMAINING_ARGS[@]}"
