#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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
CONDA_ENV_NAME="${VLLM_CONDA_ENV:-}"
if [[ -z "${CONDA_ENV_NAME}" ]]; then
  CONDA_ENV_NAME="${DEFAULT_VLLM_CONDA_ENV:-qwen}"
fi
if [[ "${CONDA_ENV_NAME}" == "brj" ]]; then
  echo "[start_vllm] refusing to start vLLM in brj env; please use qwen" >&2
  exit 1
fi
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

CONDA_SH="${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}"
if [[ -f "${CONDA_SH}" ]]; then
  # Only the vLLM server uses qwen; the main pipeline keeps running in brj.
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV_NAME}"
else
  echo "[start_vllm] conda.sh not found: ${CONDA_SH}" >&2
  exit 1
fi
echo "[start_vllm] using conda env: ${CONDA_ENV_NAME}" >&2

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

CMD=(python -m vllm.entrypoints.openai.api_server
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
