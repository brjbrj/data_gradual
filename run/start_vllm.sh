#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/run/common_env.sh"
if [[ "${PIPELINE_CONFIG_LOADED:-0}" != "1" ]]; then
  load_pipeline_config "${ROOT_DIR}"
fi

BACKGROUND=0
DRY_RUN=0
MODEL_OVERRIDE=""
PID_FILE="${VLLM_PID_FILE:-${ROOT_DIR}/outputs/runtime/vllm.pid}"
LOG_FILE="${VLLM_LOG_FILE:-${ROOT_DIR}/outputs/runtime/vllm.log}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --background)
      BACKGROUND=1
      shift
      ;;
    --pid-file)
      PID_FILE="$2"
      shift 2
      ;;
    --log-file)
      LOG_FILE="$2"
      shift 2
      ;;
    --model)
      MODEL_OVERRIDE="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      break
      ;;
  esac
done

MODEL_NAME="${MODEL_OVERRIDE:-${VLLM_MODEL:-${OPENAI_MODEL:-/root/brjverl/models/Qwen3.6-27B}}}"
MODEL_NAME="$(printf '%s' "${MODEL_NAME}" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
CONDA_ENV_NAME="${VLLM_CONDA_ENV:-${DEFAULT_VLLM_CONDA_ENV:-qwen}}"
VLLM_PYTHON_BIN="${VLLM_PYTHON:-}"
API_KEY="${VLLM_API_KEY:-EMPTY}"
HOST="${VLLM_HOST-0.0.0.0}"
# VLLM_PORT is also consumed internally by older vLLM releases. Prefer the
# project-specific name while retaining the old setting as a compatibility
# fallback, then remove both variables before launching the server.
PORT="${VLLM_API_PORT:-${VLLM_PORT:-8911}}"
TP="${VLLM_TP:-2}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN-8192}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION-0.90}"
ENABLE_AUTO_TOOL_CHOICE="${VLLM_ENABLE_AUTO_TOOL_CHOICE:-1}"
TOOL_CALL_PARSER="${VLLM_TOOL_CALL_PARSER:-hermes}"
ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-0}"
FOREGROUND_LOG="${VLLM_FOREGROUND_LOG:-1}"
LOG_APPEND="${VLLM_LOG_APPEND:-0}"

CUDA_VISIBLE_DEVICES_VALUE="${VLLM_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1}}"
NCCL_P2P_DISABLE_VALUE="${VLLM_NCCL_P2P_DISABLE:-${NCCL_P2P_DISABLE:-1}}"
NCCL_IB_DISABLE_VALUE="${VLLM_NCCL_IB_DISABLE:-${NCCL_IB_DISABLE:-1}}"
NCCL_DEBUG_VALUE="${VLLM_NCCL_DEBUG:-${NCCL_DEBUG:-INFO}}"
NCCL_SOCKET_IFNAME_VALUE="${VLLM_NCCL_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME:-lo}}"
NCCL_BLOCKING_WAIT_VALUE="${VLLM_NCCL_BLOCKING_WAIT:-${NCCL_BLOCKING_WAIT:-1}}"
TORCH_NCCL_BLOCKING_WAIT_VALUE="${VLLM_TORCH_NCCL_BLOCKING_WAIT:-inherit}"
NCCL_ALGO_VALUE="${VLLM_NCCL_ALGO:-inherit}"
NCCL_P2P_LEVEL_VALUE="${VLLM_NCCL_P2P_LEVEL:-inherit}"
NCCL_PXN_DISABLE_VALUE="${VLLM_NCCL_PXN_DISABLE:-inherit}"
NCCL_CUMEM_ENABLE_VALUE="${VLLM_NCCL_CUMEM_ENABLE:-inherit}"
GLOO_SOCKET_IFNAME_VALUE="${VLLM_GLOO_SOCKET_IFNAME:-inherit}"
VLLM_HOST_IP_VALUE="${VLLM_HOST_IP_CONFIG:-inherit}"
VLLM_ATTENTION_BACKEND_VALUE="${VLLM_ATTENTION_BACKEND:-inherit}"
VLLM_USE_FLASH_ATTN_VALUE="${VLLM_USE_FLASH_ATTN:-inherit}"
VLLM_USE_FLASHINFER_VALUE="${VLLM_USE_FLASHINFER:-inherit}"
FLASHINFER_DISABLE_JIT_VALUE="${FLASHINFER_DISABLE_JIT:-inherit}"

is_enabled() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

apply_env_setting() {
  local variable_name="$1"
  local value="$2"
  case "${value,,}" in
    unset)
      unset "${variable_name}"
      echo "[start_vllm] unset ${variable_name}" >&2
      ;;
    inherit)
      local current_value="<unset>"
      if [[ -v "${variable_name}" ]]; then
        current_value="${!variable_name}"
      fi
      echo "[start_vllm] inherit ${variable_name}=${current_value}" >&2
      ;;
    *)
      export "${variable_name}=${value}"
      echo "[start_vllm] export ${variable_name}=${value}" >&2
      ;;
  esac
}

if [[ -n "${VLLM_PYTHON_BIN}" ]]; then
  if [[ ! -x "${VLLM_PYTHON_BIN}" ]]; then
    echo "[start_vllm] VLLM_PYTHON is not executable: ${VLLM_PYTHON_BIN}" >&2
    exit 1
  fi
  VLLM_ENV_PREFIX="$(cd "$(dirname "${VLLM_PYTHON_BIN}")/.." && pwd)"
  export CONDA_PREFIX="${VLLM_ENV_PREFIX}"
  export CONDA_DEFAULT_ENV="$(basename "${VLLM_ENV_PREFIX}")"
  export PATH="${VLLM_ENV_PREFIX}/bin:${PATH}"
  export PYTHONNOUSERSITE=1
  unset PYTHONHOME PYTHONPATH PYTHONUSERBASE VIRTUAL_ENV
  echo "[start_vllm] using configured Python: ${VLLM_PYTHON_BIN}" >&2
  echo "[start_vllm] configured Python env prefix: ${VLLM_ENV_PREFIX}" >&2
else
  if [[ -z "${CONDA_ENV_NAME}" ]]; then
    VLLM_PYTHON_BIN="$(command -v python)"
    echo "[start_vllm] using current Python environment" >&2
  else
    if ! CONDA_SH_PATH="$(resolve_conda_sh)"; then
      echo "[start_vllm] cannot locate conda.sh; set CONDA_SH" >&2
      exit 1
    fi
    # shellcheck disable=SC1090
    source "${CONDA_SH_PATH}"
    conda activate "${CONDA_ENV_NAME}"
    VLLM_PYTHON_BIN="$(command -v python)"
    echo "[start_vllm] using conda environment: ${CONDA_ENV_NAME}" >&2
  fi
fi

apply_env_setting "CUDA_VISIBLE_DEVICES" "${CUDA_VISIBLE_DEVICES_VALUE}"
apply_env_setting "NCCL_P2P_DISABLE" "${NCCL_P2P_DISABLE_VALUE}"
apply_env_setting "NCCL_IB_DISABLE" "${NCCL_IB_DISABLE_VALUE}"
apply_env_setting "NCCL_DEBUG" "${NCCL_DEBUG_VALUE}"
apply_env_setting "NCCL_SOCKET_IFNAME" "${NCCL_SOCKET_IFNAME_VALUE}"
apply_env_setting "NCCL_BLOCKING_WAIT" "${NCCL_BLOCKING_WAIT_VALUE}"
apply_env_setting "TORCH_NCCL_BLOCKING_WAIT" "${TORCH_NCCL_BLOCKING_WAIT_VALUE}"
apply_env_setting "NCCL_ALGO" "${NCCL_ALGO_VALUE}"
apply_env_setting "NCCL_P2P_LEVEL" "${NCCL_P2P_LEVEL_VALUE}"
apply_env_setting "NCCL_PXN_DISABLE" "${NCCL_PXN_DISABLE_VALUE}"
apply_env_setting "NCCL_CUMEM_ENABLE" "${NCCL_CUMEM_ENABLE_VALUE}"
apply_env_setting "GLOO_SOCKET_IFNAME" "${GLOO_SOCKET_IFNAME_VALUE}"
apply_env_setting "VLLM_HOST_IP" "${VLLM_HOST_IP_VALUE}"
apply_env_setting "VLLM_ATTENTION_BACKEND" "${VLLM_ATTENTION_BACKEND_VALUE}"
apply_env_setting "VLLM_USE_FLASH_ATTN" "${VLLM_USE_FLASH_ATTN_VALUE}"
apply_env_setting "VLLM_USE_FLASHINFER" "${VLLM_USE_FLASHINFER_VALUE}"
apply_env_setting "FLASHINFER_DISABLE_JIT" "${FLASHINFER_DISABLE_JIT_VALUE}"

mkdir -p "$(dirname "${PID_FILE}")"
mkdir -p "$(dirname "${LOG_FILE}")"
MODEL_FILE="${VLLM_MODEL_FILE:-${PID_FILE%.pid}.model}"
PYTHON_FILE="${VLLM_PYTHON_FILE:-${PID_FILE%.pid}.python}"

VLLM_VERSION_INFO="$("${VLLM_PYTHON_BIN}" -c 'import sys, vllm; print("executable=%s" % sys.executable); print("vllm_version=%s" % getattr(vllm, "__version__", "unknown"))')"
echo "[start_vllm] python/vllm probe:" >&2
printf '%s\n' "${VLLM_VERSION_INFO}" | sed 's/^/[start_vllm]   /' >&2
if [[ -n "${VLLM_EXPECTED_VERSION:-}" ]]; then
  DETECTED_VLLM_VERSION="$(printf '%s\n' "${VLLM_VERSION_INFO}" | awk -F= '/^vllm_version=/{print $2}')"
  if [[ "${DETECTED_VLLM_VERSION}" != "${VLLM_EXPECTED_VERSION}" ]]; then
    echo "[start_vllm] expected vLLM ${VLLM_EXPECTED_VERSION}, got ${DETECTED_VLLM_VERSION}" >&2
    exit 1
  fi
fi

CMD=("${VLLM_PYTHON_BIN}" -m vllm.entrypoints.openai.api_server
  --model "${MODEL_NAME}"
  --port "${PORT}"
  --tensor-parallel-size "${TP}"
  --trust-remote-code
)

if [[ -n "${HOST}" ]]; then
  CMD+=(--host "${HOST}")
fi
if [[ -n "${MAX_MODEL_LEN}" ]]; then
  CMD+=(--max-model-len "${MAX_MODEL_LEN}")
fi
if [[ -n "${GPU_MEMORY_UTILIZATION}" ]]; then
  CMD+=(--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}")
fi

if [[ -n "${API_KEY}" ]]; then
  CMD+=(--api-key "${API_KEY}")
fi
if is_enabled "${ENABLE_AUTO_TOOL_CHOICE}"; then
  CMD+=(--enable-auto-tool-choice)
  if [[ -n "${TOOL_CALL_PARSER}" ]]; then
    CMD+=(--tool-call-parser "${TOOL_CALL_PARSER}")
  fi
fi
if is_enabled "${ENFORCE_EAGER}"; then
  CMD+=(--enforce-eager)
fi
if is_enabled "${DISABLE_CUSTOM_ALL_REDUCE}"; then
  CMD+=(--disable-custom-all-reduce)
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf '[start_vllm] command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

# These names configure this project's launcher. Older vLLM releases also use
# some of them internally (notably VLLM_PORT), which can make the engine select
# an unexpected worker port. All effective server values are already carried
# by explicit CLI flags above.
unset \
  VLLM_MODEL \
  VLLM_BASE_URL \
  VLLM_API_KEY \
  VLLM_RUNTIME_MODE \
  VLLM_START_TIMEOUT \
  VLLM_START_POLL_SEC \
  VLLM_EXTERNAL_WAIT_TIMEOUT \
  VLLM_EXTERNAL_POLL_SEC \
  VLLM_LOG_FILE \
  VLLM_FOREGROUND_LOG \
  VLLM_LOG_APPEND \
  VLLM_HOST \
  VLLM_API_PORT \
  VLLM_PORT \
  VLLM_TP \
  VLLM_MAX_MODEL_LEN \
  VLLM_GPU_MEMORY_UTILIZATION \
  VLLM_TIMEOUT \
  VLLM_MAX_RETRIES \
  VLLM_CONDA_ENV \
  VLLM_PYTHON \
  VLLM_ENABLE_AUTO_TOOL_CHOICE \
  VLLM_TOOL_CALL_PARSER \
  VLLM_ENFORCE_EAGER \
  VLLM_DISABLE_CUSTOM_ALL_REDUCE \
  VLLM_CUDA_VISIBLE_DEVICES \
  VLLM_NCCL_P2P_DISABLE \
  VLLM_NCCL_IB_DISABLE \
  VLLM_NCCL_DEBUG \
  VLLM_NCCL_SOCKET_IFNAME \
  VLLM_NCCL_BLOCKING_WAIT \
  VLLM_TORCH_NCCL_BLOCKING_WAIT \
  VLLM_NCCL_ALGO \
  VLLM_NCCL_P2P_LEVEL \
  VLLM_NCCL_PXN_DISABLE \
  VLLM_NCCL_CUMEM_ENABLE \
  VLLM_GLOO_SOCKET_IFNAME \
  VLLM_HOST_IP_CONFIG
unset VLLM_PID_FILE VLLM_MODEL_FILE
unset VLLM_PYTHON_FILE

if [[ -f "${PID_FILE}" ]]; then
  "${ROOT_DIR}/run/stop_vllm.sh" --pid-file "${PID_FILE}"
elif pgrep -f "vllm.entrypoints.openai.api_server" >/dev/null 2>&1; then
  echo "[start_vllm] an unmanaged vLLM API server is already running." >&2
  echo "[start_vllm] stop it explicitly before starting another model." >&2
  exit 1
fi

if [[ "${BACKGROUND}" -eq 1 ]]; then
  # A dedicated session lets stop_vllm terminate the API server and every
  # multiprocessing worker as one process group during model switches.
  {
    echo "[start_vllm] using configured Python: ${VLLM_PYTHON_BIN}"
    echo "[start_vllm] configured Python env prefix: ${CONDA_PREFIX:-<unset>}"
    printf '%s\n' "${VLLM_VERSION_INFO}" | sed 's/^/[start_vllm]   /'
  } >"${LOG_FILE}"
  if command -v setsid >/dev/null 2>&1; then
    nohup setsid "${CMD[@]}" >>"${LOG_FILE}" 2>&1 &
    SERVER_PID=$!
    echo "${SERVER_PID}" > "${PID_FILE}.pgid"
  else
    nohup "${CMD[@]}" >>"${LOG_FILE}" 2>&1 &
    SERVER_PID=$!
    rm -f "${PID_FILE}.pgid"
  fi
  echo "${SERVER_PID}" > "${PID_FILE}"
  printf '%s\n' "${MODEL_NAME}" > "${MODEL_FILE}"
  printf '%s\n' "${VLLM_PYTHON_BIN}" > "${PYTHON_FILE}"
  echo "${PID_FILE}"
  exit 0
fi

if is_enabled "${FOREGROUND_LOG}"; then
  echo "[start_vllm] foreground log: ${LOG_FILE}" >&2
  set +e
  if ! is_enabled "${LOG_APPEND}"; then
    : > "${LOG_FILE}"
  fi
  if command -v setsid >/dev/null 2>&1; then
    setsid "${CMD[@]}" >>"${LOG_FILE}" 2>&1 &
    SERVER_PID=$!
    echo "${SERVER_PID}" > "${PID_FILE}.pgid"
  else
    "${CMD[@]}" >>"${LOG_FILE}" 2>&1 &
    SERVER_PID=$!
    rm -f "${PID_FILE}.pgid"
  fi
  echo "${SERVER_PID}" > "${PID_FILE}"
  printf '%s\n' "${MODEL_NAME}" > "${MODEL_FILE}"
  printf '%s\n' "${VLLM_PYTHON_BIN}" > "${PYTHON_FILE}"

  if is_enabled "${LOG_APPEND}"; then
    tail -n 0 -F "${LOG_FILE}" &
  else
    tail -n +1 -F "${LOG_FILE}" &
  fi
  TAIL_PID=$!

  cleanup_foreground() {
    local status="${1:-$?}"
    trap - INT TERM EXIT
    kill "${TAIL_PID}" >/dev/null 2>&1 || true
    "${ROOT_DIR}/run/stop_vllm.sh" --pid-file "${PID_FILE}" >/dev/null 2>&1 || true
    exit "${status}"
  }

  trap 'cleanup_foreground 130' INT
  trap 'cleanup_foreground 143' TERM
  trap 'cleanup_foreground $?' EXIT

  wait "${SERVER_PID}"
  STATUS=$?
  trap - INT TERM EXIT
  kill "${TAIL_PID}" >/dev/null 2>&1 || true
  "${ROOT_DIR}/run/stop_vllm.sh" --pid-file "${PID_FILE}" >/dev/null 2>&1 || true
  set -e
  exit "${STATUS}"
fi

exec "${CMD[@]}"
