#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/run/stage_common.sh"
stage_init "$@"

STEP_MODEL_NAME="${STEP_MODEL:-${JUDGE_MODEL:-${VLLM_JUDGE_MODEL:-${VLLM_MODEL:-}}}}"

stage_require_file "${KB_RECORDS_PATH}" "run: bash run/01_build_kb.sh ${DATASET_NAME}"
stage_require_file "${VICTIM_ANSWER_RAW_PATH}" "run: bash run/02_answer_seed.sh ${DATASET_NAME}"
stage_ensure_vllm "${STEP_MODEL_NAME}" "step scoring"

stage_log "03 score_seed input=${VICTIM_ANSWER_RAW_PATH} output=${STEP_EVALUATION_PATH}"
"${PYTHON_BIN}" "${ROOT_DIR}/run/score_seed_answers.py" \
  --input "${VICTIM_ANSWER_RAW_PATH}" \
  --seed-input "${KB_RECORDS_PATH}" \
  --output-dir "${ANALYSIS_DIR}" \
  --step-output "${STEP_EVALUATION_PATH}" \
  --mastery-record-output "${MASTERY_PATH}" \
  --mastery-output "${MASTERY_JSON_PATH}" \
  --synthesis-target-multiplier "${SYNTHESIS_TARGET_MULTIPLIER:-26}" \
  --synthesis-min-per-seed "${SYNTHESIS_MIN_PER_SEED:-10}" \
  --synthesis-max-per-seed "${SYNTHESIS_MAX_PER_SEED:-50}" \
  --synthesis-balance-lambda "${SYNTHESIS_BALANCE_LAMBDA:-0.3}" \
  "${STAGE_REMAINING_ARGS[@]}"
