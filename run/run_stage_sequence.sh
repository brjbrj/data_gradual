#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ARG="${1:-${DATASET_NAME:-gsm8k}}"

echo "[stage-sequence] dataset=${DATASET_ARG}"
echo "[stage-sequence] STAGE_VLLM_MODE=${STAGE_VLLM_MODE:-external}"
echo "[stage-sequence] If different stages use different models, start/switch vLLM before the corresponding numbered script."

bash "${ROOT_DIR}/run/01_build_kb.sh" "${DATASET_ARG}"
bash "${ROOT_DIR}/run/02_answer_seed.sh" "${DATASET_ARG}"
bash "${ROOT_DIR}/run/03_score_seed.sh" "${DATASET_ARG}"
bash "${ROOT_DIR}/run/04_build_synthesis_plan.sh" "${DATASET_ARG}"
bash "${ROOT_DIR}/run/05_generate_questions.sh" "${DATASET_ARG}"
if [[ "${RUN_VALIDATION:-1}" != "0" ]]; then
  bash "${ROOT_DIR}/run/06_validate_generated.sh" "${DATASET_ARG}"
  bash "${ROOT_DIR}/run/07_export_training_data.sh" "${DATASET_ARG}"
fi
