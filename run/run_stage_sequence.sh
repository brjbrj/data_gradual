#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/run/common_env.sh"
USER_STAGE_VLLM_MODE="${STAGE_VLLM_MODE-}"
MODE_WAS_SET=0
if [[ -n "${STAGE_VLLM_MODE+x}" ]]; then
  MODE_WAS_SET=1
fi
load_pipeline_config "${ROOT_DIR}"
DATASET_ARG="${1:-${DATASET_NAME:-gsm8k}}"
if [[ "${MODE_WAS_SET}" -eq 1 ]]; then
  export STAGE_VLLM_MODE="${USER_STAGE_VLLM_MODE}"
else
  export STAGE_VLLM_MODE=managed
fi
export STAGE_SEQUENCE_VLLM_MODE="${STAGE_VLLM_MODE}"
if [[ "${STAGE_VLLM_MODE}" == "managed" ]]; then
  export STAGE_VLLM_STOP_ON_EXIT="${STAGE_VLLM_STOP_ON_EXIT:-0}"
  STAGE_SEQUENCE_PID_FILE="${VLLM_PID_FILE:-${OUTPUT_DIR:-${ROOT_DIR}/outputs}/runtime/vllm/vllm.pid}"
  cleanup_sequence_vllm() {
    bash "${ROOT_DIR}/run/stop_vllm.sh" --pid-file "${STAGE_SEQUENCE_PID_FILE}" >/dev/null 2>&1 || true
  }
  trap cleanup_sequence_vllm EXIT INT TERM
fi

echo "[stage-sequence] dataset=${DATASET_ARG}"
echo "[stage-sequence] STAGE_VLLM_MODE=${STAGE_VLLM_MODE:-external}"
if [[ "${STAGE_VLLM_MODE}" == "managed" ]]; then
  echo "[stage-sequence] managed mode: stages will start/switch vLLM automatically and stop it when the sequence exits."
  if [[ "${MODE_WAS_SET}" -eq 0 ]]; then
    echo "[stage-sequence] set STAGE_VLLM_MODE=external to use a manually started vLLM server instead."
  fi
else
  echo "[stage-sequence] external mode: start/switch vLLM before each model-dependent stage."
fi

bash "${ROOT_DIR}/run/01_build_kb.sh" "${DATASET_ARG}"
bash "${ROOT_DIR}/run/02_answer_seed.sh" "${DATASET_ARG}"
bash "${ROOT_DIR}/run/03_score_seed.sh" "${DATASET_ARG}"
bash "${ROOT_DIR}/run/04_build_synthesis_plan.sh" "${DATASET_ARG}"
bash "${ROOT_DIR}/run/05_generate_questions.sh" "${DATASET_ARG}"
if [[ "${RUN_VALIDATION:-1}" != "0" ]]; then
  bash "${ROOT_DIR}/run/06_validate_generated.sh" "${DATASET_ARG}"
  # Optional final wording pass: keep validated math fixed, polish only steps
  # for training friendliness. Disable with RUN_STEP_REFINEMENT=0.
  if [[ "${RUN_STEP_REFINEMENT:-1}" != "0" ]]; then
    bash "${ROOT_DIR}/run/07_refine_solution_steps.sh" "${DATASET_ARG}"
  fi
  # Export automatically uses refined.jsonl when present, otherwise validated.jsonl.
  bash "${ROOT_DIR}/run/08_export_training_data.sh" "${DATASET_ARG}"
fi
