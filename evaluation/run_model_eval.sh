#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/run/stage_common.sh"
stage_init "$@"

load_eval_config_defaults() {
  local file line key value
  for file in "${ROOT_DIR}/evaluation/eval.env" "${ROOT_DIR}/evaluation/eval.example.env"; do
    [[ -f "${file}" ]] || continue
    while IFS= read -r line || [[ -n "${line}" ]]; do
      line="${line%$'\r'}"
      [[ -n "${line}" ]] || continue
      [[ "${line}" == \#* ]] && continue
      line="${line#export }"
      [[ "${line}" == *"="* ]] || continue
      key="${line%%=*}"
      value="${line#*=}"
      key="${key//[[:space:]]/}"
      [[ "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
      if [[ -z "${!key+x}" ]]; then
        value="${value%\"}"
        value="${value#\"}"
        value="${value%\'}"
        value="${value#\'}"
        export "${key}=${value}"
      fi
    done < "${file}"
  done
}

load_eval_config_defaults

if [[ -z "${STAGE_VLLM_MODE+x}" && -n "${VLLM_RUNTIME_MODE:-}" ]]; then
  export STAGE_VLLM_MODE="${VLLM_RUNTIME_MODE}"
fi

EVAL_MODEL_NAME="${EVAL_MODEL:-${VICTIM_MODEL:-${VLLM_MODEL:-}}}"
EVAL_INPUT="${EVAL_INPUT_PATH:-${INPUT_PATH}}"
EVAL_DIR="${EVAL_OUTPUT_DIR:-${OUTPUT_DIR}/model_eval/${DATASET_NAME}}"
EVAL_PROMPT="${EVAL_PROMPT_PATH:-${ROOT_DIR}/evaluation/prompt/generate.json}"

stage_ensure_vllm "${EVAL_MODEL_NAME}" "model evaluation"

ARGS=("${PYTHON_BIN}" "${ROOT_DIR}/evaluation/evaluate_accuracy.py"
  --input "${EVAL_INPUT}"
  --output-dir "${EVAL_DIR}"
  --dataset-name "${DATASET_NAME}"
  --format-template "${EVAL_FORMAT_TEMPLATE:-auto}"
  --prompt "${EVAL_PROMPT}"
  --model "${EVAL_MODEL_NAME}"
  --base-url "${EVAL_BASE_URL:-${VLLM_BASE_URL:-}}"
  --api-key "${EVAL_API_KEY:-${VLLM_API_KEY:-EMPTY}}"
  --n-answers "${EVAL_N_ANSWERS:-1}"
  --concurrency "${EVAL_CONCURRENCY:-64}"
  --temperature "${EVAL_TEMPERATURE:-0.0}"
  --top-p "${EVAL_TOP_P:-0.95}"
  --max-tokens "${EVAL_MAX_TOKENS:-1500}"
  --presence-penalty "${EVAL_PRESENCE_PENALTY:-0.0}"
  --frequency-penalty "${EVAL_FREQUENCY_PENALTY:-0.0}"
  --prompt-mode "${EVAL_PROMPT_MODE:-legacy_concat}"
  --answer-extract-mode "${EVAL_ANSWER_EXTRACT_MODE:-number}"
  --timeout "${EVAL_TIMEOUT:-600}"
  --max-retries "${EVAL_MAX_RETRIES:--1}"
)

if [[ -n "${EVAL_SEED_BASE:-}" ]]; then
  ARGS+=(--seed-base "${EVAL_SEED_BASE}")
fi
if stage_truthy "${EVAL_ATTEMPT_VARIATION:-0}"; then
  ARGS+=(--attempt-variation)
fi

if [[ -n "${SAMPLE_LIMIT:-}" ]]; then
  ARGS+=(--sample-limit "${SAMPLE_LIMIT}")
fi
if stage_truthy "${STAGE_FORCE:-0}"; then
  ARGS+=(--no-resume)
elif stage_truthy "${EVAL_RESUME:-1}"; then
  ARGS+=(--resume)
fi
ARGS+=("${STAGE_REMAINING_ARGS[@]}")

stage_log "model_eval input=${EVAL_INPUT} output_dir=${EVAL_DIR} model=${EVAL_MODEL_NAME} n=${EVAL_N_ANSWERS:-1}"
"${ARGS[@]}"
