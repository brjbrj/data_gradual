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
      echo "[stop_vllm] unknown argument: $1" >&2
      exit 2
      ;;
  esac
done
SELF_PGID="$(ps -o pgid= -p "$$" 2>/dev/null | tr -d '[:space:]' || true)"

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

pid_matches_port() {
  local pid="$1"
  local expected_port="$2"
  [[ -z "${expected_port}" ]] && return 0
  local cmdline
  cmdline="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
  [[ "${cmdline}" == *"--port ${expected_port}"* || "${cmdline}" == *"--port=${expected_port}"* ]]
}

cleanup_metadata() {
  local pid_file="$1"
  rm -f \
    "${pid_file}" \
    "${pid_file}.pgid" \
    "${pid_file%.pid}.model" \
    "${pid_file%.pid}.python" \
    "${pid_file%.pid}.port"
}

stop_pid() {
  local pid="$1"
  local pid_file="${2:-}"
  if ! kill -0 "${pid}" >/dev/null 2>&1; then
    [[ -n "${pid_file}" ]] && cleanup_metadata "${pid_file}"
    return 0
  fi
  CMDLINE="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
  if [[ "${CMDLINE}" != *"vllm.entrypoints.openai.api_server"* ]]; then
    echo "[stop_vllm] PID ${pid} is not a vLLM API server; skipped" >&2
    [[ -n "${pid_file}" ]] && cleanup_metadata "${pid_file}"
    return 0
  fi
  if ! pid_matches_port "${pid}" "${PORT}"; then
    echo "[stop_vllm] PID ${pid} is vLLM but not on port ${PORT}; skipped" >&2
    return 0
  fi

  PGID=""
  if [[ -n "${pid_file}" && -f "${pid_file}.pgid" ]]; then
    PGID="$(cat "${pid_file}.pgid")"
  fi
  if [[ -z "${PGID}" ]]; then
    PGID="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d '[:space:]' || true)"
  fi

  if [[ -n "${PGID}" && "${PGID}" != "${SELF_PGID}" ]]; then
    kill -TERM -- "-${PGID}" >/dev/null 2>&1 || true
    if ! wait_for_group_exit "${PGID}" 30; then
      kill -KILL -- "-${PGID}" >/dev/null 2>&1 || true
      wait_for_group_exit "${PGID}" 10 || true
    fi
  else
    signal_process_tree TERM "${pid}"
    if ! wait_for_exit "${pid}" 30; then
      signal_process_tree KILL "${pid}"
    fi
  fi
  [[ -n "${pid_file}" ]] && cleanup_metadata "${pid_file}"
}

find_vllm_pids_by_port() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -tiTCP:"${port}" -sTCP:LISTEN 2>/dev/null || true
    return
  fi
  if command -v fuser >/dev/null 2>&1; then
    fuser "${port}/tcp" 2>/dev/null | tr ' ' '\n' || true
    return
  fi
  if command -v ss >/dev/null 2>&1; then
    ss -ltnp "sport = :${port}" 2>/dev/null \
      | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' \
      | sort -u || true
    return
  fi
}

if [[ -n "${PID_FILE}" && -f "${PID_FILE}" ]]; then
  PID="$(cat "${PID_FILE}")"
  stop_pid "${PID}" "${PID_FILE}"
  if [[ -n "${PORT}" ]]; then
    while read -r PID; do
      [[ -z "${PID}" || "${PID}" == "$$" ]] && continue
      stop_pid "${PID}" ""
    done < <(find_vllm_pids_by_port "${PORT}")
  fi
elif [[ -n "${PORT}" ]]; then
  FOUND=0
  while read -r PID; do
    [[ -z "${PID}" || "${PID}" == "$$" ]] && continue
    FOUND=1
    stop_pid "${PID}" ""
  done < <(find_vllm_pids_by_port "${PORT}")
  if [[ "${FOUND}" -eq 0 ]]; then
    echo "[stop_vllm] no vLLM listener found on port ${PORT}" >&2
  fi
else
  echo "[stop_vllm] refusing to stop vLLM without --pid-file or --port" >&2
  exit 2
fi
