#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/run/common_env.sh"
if [[ "${PIPELINE_CONFIG_LOADED:-0}" != "1" ]]; then
  load_pipeline_config "${ROOT_DIR}"
fi

BACKGROUND=0
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
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0,1}"
NCCL_P2P_DISABLE_VALUE="${NCCL_P2P_DISABLE:-1}"
NCCL_IB_DISABLE_VALUE="${NCCL_IB_DISABLE:-1}"
NCCL_DEBUG_VALUE="${NCCL_DEBUG:-INFO}"
NCCL_SOCKET_IFNAME_VALUE="${NCCL_SOCKET_IFNAME:-lo}"
NCCL_BLOCKING_WAIT_VALUE="${NCCL_BLOCKING_WAIT:-1}"

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

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE_VALUE}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE_VALUE}"
export NCCL_DEBUG="${NCCL_DEBUG_VALUE}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME_VALUE}"
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT_VALUE}"

pkill -f "vllm.entrypoints.openai.api_server" >/dev/null 2>&1 || true

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
  --enable-auto-tool-choice
  --tool-call-parser hermes
)

if [[ "${BACKGROUND}" -eq 1 ]]; then
  nohup "${CMD[@]}" >"${LOG_FILE}" 2>&1 &
  echo $! > "${PID_FILE}"
  printf '%s\n' "${MODEL_NAME}" > "${MODEL_FILE}"
  echo "${PID_FILE}"
  exit 0
fi

exec "${CMD[@]}"
