#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/run/stage_common.sh"
stage_init "$@"

QC_MODEL_NAME="${QC_MODEL:-${QUALITY_MODEL:-${STEP_MODEL:-${VLLM_MODEL:-}}}}"
GEN_MODEL_NAME="${GEN_MODEL:-${VLLM_GEN_MODEL:-${VLLM_MODEL:-}}}"

stage_require_file "${GENERATED_OUTPUT_PATH}" "run: bash run/05_generate_questions.sh ${DATASET_NAME}"
stage_require_file "${PLAN_PATH}" "run: bash run/04_build_synthesis_plan.sh ${DATASET_NAME}"
stage_require_file "${MASTERY_PATH}" "run: bash run/03_score_seed.sh ${DATASET_NAME}"
stage_ensure_vllm "${QC_MODEL_NAME}" "validation"

jsonl_count() {
  local path="$1"
  "${PYTHON_BIN}" -c 'import sys; from pathlib import Path; p=Path(sys.argv[1]); print(sum(1 for line in p.open("r", encoding="utf-8") if line.strip()) if p.exists() else 0)' "${path}"
}

if stage_skip_if_complete "06_validate_generated" "${VALIDATED_OUTPUT_PATH}" "${VALIDATION_REPORTS_PATH}"; then
  if ! stage_truthy "${VALIDATION_BACKTRACK_ON_COMPLETE:-1}"; then
    exit 0
  fi
  if [[ ! -s "${VALIDATION_FAILED_PATH}" ]] || [[ "$(jsonl_count "${VALIDATION_FAILED_PATH}")" -eq 0 ]]; then
    exit 0
  fi
  stage_log "existing validation has failures; continuing validation backtrack loop"
fi

run_validation_once() {
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
}

run_generate_retry_once() {
  stage_log "convert validation failures to generate retry input=${FAILED_OUTPUT_PATH}"
  "${PYTHON_BIN}" "${ROOT_DIR}/run/validation_failures_to_generate_retry.py" \
    --validation-failed "${VALIDATION_FAILED_PATH}" \
    --plan "${PLAN_PATH}" \
    --output "${FAILED_OUTPUT_PATH}"

  local retry_count
  retry_count="$(jsonl_count "${FAILED_OUTPUT_PATH}")"
  if [[ "${retry_count}" -eq 0 ]]; then
    stage_log "no generate retry records converted from validation failures"
    return 1
  fi

  if ! stage_models_match "${QC_MODEL_NAME}" "${GEN_MODEL_NAME}"; then
    stage_log "validation backtrack needs generation model; checking vLLM switch"
    stage_ensure_vllm "${GEN_MODEL_NAME}" "validation backtrack generation"
  else
    stage_log "validation and generation models match; reusing current vLLM"
  fi

  stage_log "05 generate_questions retry_from_validation count=${retry_count}"
  GEN_RETRY_COMPLETED_FAILURES=1 "${PYTHON_BIN}" "${ROOT_DIR}/run/generate_questions.py" \
    --plan "${PLAN_PATH}" \
    --mastery "${MASTERY_PATH}" \
    --output "${GENERATED_OUTPUT_PATH}" \
    --raw-output "${RAW_OUTPUT_PATH}" \
    --failed-output "${FAILED_OUTPUT_PATH}" \
    --model "${GEN_MODEL_NAME}" \
    --checkpoint-every "${GEN_CHECKPOINT_EVERY:-50}"

  if ! stage_models_match "${QC_MODEL_NAME}" "${GEN_MODEL_NAME}"; then
    stage_log "switching vLLM back to validation model"
    stage_ensure_vllm "${QC_MODEL_NAME}" "validation"
  fi
}

BACKTRACK_MAX_ROUNDS="${VALIDATION_BACKTRACK_MAX_ROUNDS:-3}"
BACKTRACK_ROUND=0

while true; do
  run_validation_once
  FAILED_COUNT="$(jsonl_count "${VALIDATION_FAILED_PATH}")"
  stage_log "validation failed_count=${FAILED_COUNT} backtrack_round=${BACKTRACK_ROUND}/${BACKTRACK_MAX_ROUNDS}"
  if [[ "${FAILED_COUNT}" -eq 0 ]]; then
    exit 0
  fi
  if [[ "${BACKTRACK_MAX_ROUNDS}" -ge 0 && "${BACKTRACK_ROUND}" -ge "${BACKTRACK_MAX_ROUNDS}" ]]; then
    stage_log "validation backtrack limit reached; remaining failures preserved in ${VALIDATION_FAILED_PATH}"
    exit 0
  fi
  if ! run_generate_retry_once; then
    stage_log "validation backtrack could not create generation retries; remaining failures preserved in ${VALIDATION_FAILED_PATH}"
    exit 0
  fi
  BACKTRACK_ROUND=$((BACKTRACK_ROUND + 1))
done
