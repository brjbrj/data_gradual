#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/run/common_env.sh"
load_pipeline_config "${ROOT_DIR}"
activate_pipeline_env
PYTHON_BIN="$(resolve_pipeline_python)"

DATASET_NAME="${DATASET_NAME:-gsm8k}"
if [[ $# -gt 0 && "${1:-}" != --* ]]; then
  DATASET_NAME="$1"
  shift
fi

PIPELINE_OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs}"
PLAN_PATH="${PLAN_PATH:-${PIPELINE_OUTPUT_DIR}/planning/${DATASET_NAME}/synthesis_plan.jsonl}"
MASTERY_PATH="${MASTERY_PATH:-${PIPELINE_OUTPUT_DIR}/analysis/${DATASET_NAME}/mastery_records.jsonl}"
GENERATED_OUTPUT_PATH="${GENERATED_OUTPUT_PATH:-${PIPELINE_OUTPUT_DIR}/pipeline/${DATASET_NAME}/generated.jsonl}"
RAW_OUTPUT_PATH="${RAW_OUTPUT_PATH:-${PIPELINE_OUTPUT_DIR}/pipeline/${DATASET_NAME}/generated.raw.jsonl}"
FAILED_OUTPUT_PATH="${FAILED_OUTPUT_PATH:-${PIPELINE_OUTPUT_DIR}/pipeline/${DATASET_NAME}/generated.failed.jsonl}"

MANAGED_VLLM=0
VLLM_PID_FILE="${VLLM_PID_FILE:-${PIPELINE_OUTPUT_DIR}/runtime/vllm/vllm.pid}"
VLLM_LOG_PATH="${VLLM_LOG_FILE:-${PIPELINE_OUTPUT_DIR}/runtime/vllm/vllm.log}"
GEN_MODEL_NAME="${GEN_MODEL:-${VLLM_GEN_MODEL:-${VLLM_MODEL:-/root/brjverl/models/Qwen3.6-27B}}}"

normalize_model_name() {
  printf '%s' "$1" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//; s:/*$::'
}

wait_for_vllm_model() {
  local expected
  expected="$(normalize_model_name "$1")"
  local timeout="${VLLM_START_TIMEOUT:-600}"
  local poll_sec="${VLLM_START_POLL_SEC:-5}"
  local waited=0
  local running=""
  while true; do
    running="$("${PYTHON_BIN}" "${ROOT_DIR}/run/probe_vllm.py" 2>/dev/null || true)"
    if [[ -n "${running}" && "$(normalize_model_name "${running}")" == "${expected}" ]]; then
      echo "[run_generate_questions] vLLM ready: ${running}" >&2
      return 0
    fi
    if [[ "${timeout}" -ge 0 && "${waited}" -ge "${timeout}" ]]; then
      echo "[run_generate_questions] vLLM did not become ready for ${expected} within ${timeout}s" >&2
      echo "[run_generate_questions] last probed model: ${running:-<none>}" >&2
      if [[ -f "${VLLM_LOG_PATH}" ]]; then
        echo "[run_generate_questions] last vLLM log lines:" >&2
        tail -40 "${VLLM_LOG_PATH}" >&2 || true
      fi
      return 1
    fi
    echo "[run_generate_questions] waiting for vLLM model ${expected} (${waited}s/${timeout}s)" >&2
    sleep "${poll_sec}"
    waited=$((waited + poll_sec))
  done
}

cleanup_managed_vllm() {
  local status="${1:-$?}"
  trap - INT TERM EXIT
  if [[ "${MANAGED_VLLM}" -eq 1 ]]; then
    "${ROOT_DIR}/run/stop_vllm.sh" --pid-file "${VLLM_PID_FILE}" >/dev/null 2>&1 || true
  fi
  exit "${status}"
}

if [[ "${VLLM_RUNTIME_MODE:-external}" == "managed" ]]; then
  MANAGED_VLLM=1
  trap 'cleanup_managed_vllm 130' INT
  trap 'cleanup_managed_vllm 143' TERM
  trap 'cleanup_managed_vllm $?' EXIT
  echo "[run_generate_questions] starting managed vLLM model: ${GEN_MODEL_NAME}" >&2
  "${ROOT_DIR}/run/start_vllm.sh" \
    --background \
    --pid-file "${VLLM_PID_FILE}" \
    --log-file "${VLLM_LOG_PATH}" \
    --model "${GEN_MODEL_NAME}" >/dev/null
  wait_for_vllm_model "${GEN_MODEL_NAME}"
else
  echo "[run_generate_questions] external vLLM mode; expecting ${GEN_MODEL_NAME} at ${VLLM_BASE_URL:-http://127.0.0.1:8911/v1}" >&2
fi

"${PYTHON_BIN}" "${ROOT_DIR}/run/generate_questions.py" \
  --plan "${PLAN_PATH}" \
  --mastery "${MASTERY_PATH}" \
  --output "${GENERATED_OUTPUT_PATH}" \
  --raw-output "${RAW_OUTPUT_PATH}" \
  --failed-output "${FAILED_OUTPUT_PATH}" \
  "$@"

STATUS=$?
cleanup_managed_vllm "${STATUS}"
