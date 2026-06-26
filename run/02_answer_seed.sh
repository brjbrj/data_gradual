#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/run/stage_common.sh"
stage_init "$@"

VICTIM_MODEL_NAME="${VICTIM_MODEL:-${VLLM_VICTIM_MODEL:-${VLLM_MODEL:-}}}"
N_ANSWERS="${N_ANSWERS:-10}"

stage_require_file "${KB_RECORDS_PATH}" "run: bash run/01_build_kb.sh ${DATASET_NAME}"
stage_ensure_vllm "${VICTIM_MODEL_NAME}" "victim answering"

stage_log "02 answer_seed input=${KB_RECORDS_PATH} output=${VICTIM_ANSWER_RAW_PATH}"
"${PYTHON_BIN}" "${ROOT_DIR}/run/answer_seed_questions.py" \
  --mode answer \
  --input "${KB_RECORDS_PATH}" \
  --output-dir "${ANALYSIS_DIR}" \
  --n-answers "${N_ANSWERS}" \
  --temperature "${VICTIM_TEMPERATURE:-0.3}" \
  --top-p "${VICTIM_TOP_P:-0.95}" \
  --answer-output "${VICTIM_ANSWER_PATH}" \
  --answer-raw-output "${VICTIM_ANSWER_RAW_PATH}" \
  "${STAGE_REMAINING_ARGS[@]}"
