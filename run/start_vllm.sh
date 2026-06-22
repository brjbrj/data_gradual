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
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      break
      ;;
  esac
done

MODEL_NAME="${VLLM_MODEL:-${OPENAI_MODEL:-/root/brjverl/models/Qwen3.6-27B}}"
CONDA_ENV_NAME="${VLLM_CONDA_ENV:-${DEFAULT_VLLM_CONDA_ENV:-qwen}}"
VLLM_PYTHON_BIN="${VLLM_PYTHON:-}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8911}"
TP="${VLLM_TP:-2}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
ENABLE_AUTO_TOOL_CHOICE="${VLLM_ENABLE_AUTO_TOOL_CHOICE:-1}"
TOOL_CALL_PARSER="${VLLM_TOOL_CALL_PARSER:-hermes}"
ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-0}"

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
  echo "[start_vllm] using configured Python: ${VLLM_PYTHON_BIN}" >&2
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

mkdir -p "$(dirname "${PID_FILE}")"
mkdir -p "$(dirname "${LOG_FILE}")"
MODEL_FILE="${VLLM_MODEL_FILE:-${PID_FILE%.pid}.model}"

CMD=("${VLLM_PYTHON_BIN}" -m vllm.entrypoints.openai.api_server
  --model "${MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --tensor-parallel-size "${TP}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --trust-remote-code
)

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

pkill -f "vllm.entrypoints.openai.api_server" >/dev/null 2>&1 || true

if [[ "${BACKGROUND}" -eq 1 ]]; then
  nohup "${CMD[@]}" >"${LOG_FILE}" 2>&1 &
  echo $! > "${PID_FILE}"
  printf '%s\n' "${MODEL_NAME}" > "${MODEL_FILE}"
  echo "${PID_FILE}"
  exit 0
fi

exec "${CMD[@]}"
