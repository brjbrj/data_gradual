#!/usr/bin/env bash
set -euo pipefail

PID_FILE=""
PORT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pid-file)
      PID_FILE="${2:-}"
      shift 2
      ;;
    --port)
      PORT="${2:-}"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done
SELF_PGID="$(ps -o pgid= -p "$$" 2>/dev/null | tr -d '[:space:]' || true)"

cmdline_has_port() {
  local cmdline="$1"
  local port="$2"
  [[ -z "${port}" ]] && return 0
  [[ " ${cmdline} " == *" --port ${port} "* || " ${cmdline} " == *" --port=${port} "* ]]
}

wait_for_exit() {
  local pid="$1"
  local attempts="${2:-30}"
  local index
  for ((index = 0; index < attempts; index++)); do
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
  done
  return 1
}

wait_for_group_exit() {
  local pgid="$1"
  local attempts="${2:-30}"
  local index
  for ((index = 0; index < attempts; index++)); do
    if ! kill -0 -- "-${pgid}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
  done
  return 1
}

signal_process_tree() {
  local signal_name="$1"
  local parent_pid="$2"
  local child_pid
  while read -r child_pid; do
    [[ -z "${child_pid}" ]] && continue
    signal_process_tree "${signal_name}" "${child_pid}"
  done < <(pgrep -P "${parent_pid}" 2>/dev/null || true)
  kill "-${signal_name}" "${parent_pid}" >/dev/null 2>&1 || true
}

stop_api_pids() {
  local pids="$1"
  [[ -z "${pids}" ]] && return 0
  while read -r pid; do
    [[ -z "${pid}" || "${pid}" == "$$" ]] && continue
    signal_process_tree TERM "${pid}"
  done <<< "${pids}"
  sleep 5
  while read -r pid; do
    [[ -z "${pid}" || "${pid}" == "$$" ]] && continue
    if kill -0 "${pid}" >/dev/null 2>&1; then
      signal_process_tree KILL "${pid}"
    fi
  done <<< "${pids}"
}

api_server_pids() {
  if [[ -n "${PORT}" ]]; then
    ps -eo pid=,args= | awk -v port="${PORT}" '
      /[v]llm\.entrypoints\.openai\.api_server/ {
        for (i = 1; i <= NF; i++) {
          if ($i == "--port" && (i + 1) <= NF && $(i + 1) == port) {
            print $1
            next
          }
          if ($i == "--port=" port) {
            print $1
            next
          }
        }
      }
    '
  elif [[ "${STOP_VLLM_ALLOW_GLOBAL:-0}" == "1" ]]; then
    pgrep -f "vllm.entrypoints.openai.api_server" || true
  else
    echo "[stop_vllm] no usable PID file and no --port; refusing global vLLM shutdown" >&2
    echo "[stop_vllm] pass --port or set STOP_VLLM_ALLOW_GLOBAL=1 to allow the old global fallback" >&2
    return 0
  fi
}

if [[ -n "${PID_FILE}" && -f "${PID_FILE}" ]]; then
  PID="$(cat "${PID_FILE}")"
  if kill -0 "${PID}" >/dev/null 2>&1; then
    CMDLINE="$(tr '\0' ' ' < "/proc/${PID}/cmdline" 2>/dev/null || true)"
    if [[ "${CMDLINE}" != *"vllm.entrypoints.openai.api_server"* ]]; then
      echo "[stop_vllm] stale PID file ignored; PID ${PID} is not vLLM" >&2
      rm -f "${PID_FILE}" "${PID_FILE}.pgid" "${PID_FILE%.pid}.model" "${PID_FILE%.pid}.python"
      API_PIDS="$(api_server_pids || true)"
      stop_api_pids "${API_PIDS}"
      exit 0
    fi
    if ! cmdline_has_port "${CMDLINE}" "${PORT}"; then
      echo "[stop_vllm] PID file points to vLLM on a different port; falling back to --port ${PORT}" >&2
      API_PIDS="$(api_server_pids || true)"
      stop_api_pids "${API_PIDS}"
      rm -f "${PID_FILE}" "${PID_FILE}.pgid" "${PID_FILE%.pid}.model" "${PID_FILE%.pid}.python"
      exit 0
    fi

    PGID=""
    ACTUAL_PGID="$(ps -o pgid= -p "${PID}" 2>/dev/null | tr -d '[:space:]' || true)"
    if [[ -f "${PID_FILE}.pgid" ]]; then
      RECORDED_PGID="$(cat "${PID_FILE}.pgid")"
      if [[ -n "${RECORDED_PGID}" && "${RECORDED_PGID}" == "${ACTUAL_PGID}" ]]; then
        PGID="${RECORDED_PGID}"
      else
        echo "[stop_vllm] stale PGID file ignored for PID ${PID}" >&2
      fi
    fi
    if [[ -z "${PGID}" ]]; then
      PGID="${ACTUAL_PGID}"
    fi

    if [[ -n "${PGID}" && "${PGID}" != "${SELF_PGID}" ]]; then
      kill -TERM -- "-${PGID}" >/dev/null 2>&1 || true
      if ! wait_for_group_exit "${PGID}" 30; then
        kill -KILL -- "-${PGID}" >/dev/null 2>&1 || true
        wait_for_group_exit "${PGID}" 10 || true
      fi
    else
      signal_process_tree TERM "${PID}"
      if ! wait_for_exit "${PID}" 30; then
        signal_process_tree KILL "${PID}"
      fi
    fi
  else
    echo "[stop_vllm] stale PID file ignored; PID ${PID} is not running" >&2
    rm -f "${PID_FILE}" "${PID_FILE}.pgid" "${PID_FILE%.pid}.model" "${PID_FILE%.pid}.python"
    API_PIDS="$(api_server_pids || true)"
    stop_api_pids "${API_PIDS}"
    exit 0
  fi
  rm -f "${PID_FILE}" "${PID_FILE}.pgid" "${PID_FILE%.pid}.model" "${PID_FILE%.pid}.python"
else
  # Compatibility fallback for servers started before process-group tracking.
  API_PIDS="$(api_server_pids || true)"
  stop_api_pids "${API_PIDS}"
fi
