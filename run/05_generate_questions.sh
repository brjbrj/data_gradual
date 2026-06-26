#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/run/stage_common.sh"
stage_init "$@"

GEN_MODEL_NAME="${GEN_MODEL:-${VLLM_GEN_MODEL:-${VLLM_MODEL:-}}}"

stage_require_file "${PLAN_PATH}" "run: bash run/04_build_synthesis_plan.sh ${DATASET_NAME}"
stage_require_file "${MASTERY_PATH}" "run: bash run/03_score_seed.sh ${DATASET_NAME}"
stage_ensure_vllm "${GEN_MODEL_NAME}" "question generation"

stage_log "05 generate_questions output=${GENERATED_OUTPUT_PATH}"
"${PYTHON_BIN}" "${ROOT_DIR}/run/generate_questions.py" \
  --plan "${PLAN_PATH}" \
  --mastery "${MASTERY_PATH}" \
  --output "${GENERATED_OUTPUT_PATH}" \
  --raw-output "${RAW_OUTPUT_PATH}" \
  --failed-output "${FAILED_OUTPUT_PATH}" \
  --model "${GEN_MODEL_NAME}" \
  "${STAGE_REMAINING_ARGS[@]}"
