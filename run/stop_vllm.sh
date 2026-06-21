#!/usr/bin/env bash
set -euo pipefail

PID_FILE=""
if [[ "${1:-}" == "--pid-file" ]]; then
  PID_FILE="${2:-}"
fi

if [[ -n "${PID_FILE}" && -f "${PID_FILE}" ]]; then
  PID="$(cat "${PID_FILE}")"
  if kill -0 "${PID}" >/dev/null 2>&1; then
    kill "${PID}" >/dev/null 2>&1 || true
    for _ in {1..20}; do
      if ! kill -0 "${PID}" >/dev/null 2>&1; then
        break
      fi
      sleep 0.5
    done
    if kill -0 "${PID}" >/dev/null 2>&1; then
      kill -9 "${PID}" >/dev/null 2>&1 || true
    fi
  fi
  rm -f "${PID_FILE}"
else
  pkill -f "vllm.entrypoints.openai.api_server" >/dev/null 2>&1 || true
fi

