#!/usr/bin/env bash
set -euo pipefail

STAGE_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${STAGE_ROOT_DIR}/run/common_env.sh"

stage_init() {
  load_pipeline_config "${STAGE_ROOT_DIR}"
  activate_pipeline_env
  PYTHON_BIN="$(resolve_pipeline_python)"
  export PYTHON_BIN

  DATASET_NAME="${DATASET_NAME:-gsm8k}"
  if [[ $# -gt 0 && "${1:-}" != --* ]]; then
    DATASET_NAME="$1"
    shift
  fi
  export DATASET_NAME

  OUTPUT_DIR="${OUTPUT_DIR:-${STAGE_ROOT_DIR}/outputs}"
  INPUT_PATH="${INPUT_PATH:-${STAGE_ROOT_DIR}/data/${DATASET_NAME}.jsonl}"
  KB_DIR="${KB_DIR:-${OUTPUT_DIR}/kb/${DATASET_NAME}}"
  ANALYSIS_DIR="${ANALYSIS_DIR:-${OUTPUT_DIR}/analysis/${DATASET_NAME}}"
  PLANNING_DIR="${PLANNING_DIR:-${OUTPUT_DIR}/planning/${DATASET_NAME}}"
  PIPELINE_DIR="${PIPELINE_DIR:-${OUTPUT_DIR}/pipeline/${DATASET_NAME}}"

  KB_RECORDS_PATH="${KB_RECORDS_PATH:-${KB_DIR}/records.jsonl}"
  KB_ENTITIES_PATH="${KB_ENTITIES_PATH:-${KB_DIR}/entities.json}"
  VICTIM_ANSWER_PATH="${VICTIM_ANSWER_PATH:-${ANALYSIS_DIR}/victim_answers.jsonl}"
  VICTIM_ANSWER_RAW_PATH="${VICTIM_ANSWER_RAW_PATH:-${ANALYSIS_DIR}/victim_answers.raw.jsonl}"
  STEP_EVALUATION_PATH="${STEP_EVALUATION_PATH:-${ANALYSIS_DIR}/step_evaluations.jsonl}"
  MASTERY_PATH="${MASTERY_PATH:-${ANALYSIS_DIR}/mastery_records.jsonl}"
  MASTERY_JSON_PATH="${MASTERY_JSON_PATH:-${ANALYSIS_DIR}/mastery.json}"
  PLAN_PATH="${PLAN_PATH:-${PLANNING_DIR}/synthesis_plan.jsonl}"
  PLAN_SUMMARY_PATH="${PLAN_SUMMARY_PATH:-${PLANNING_DIR}/synthesis_plan.summary.json}"
  GENERATED_OUTPUT_PATH="${GENERATED_OUTPUT_PATH:-${PIPELINE_DIR}/generated.jsonl}"
  RAW_OUTPUT_PATH="${RAW_OUTPUT_PATH:-${PIPELINE_DIR}/generated.raw.jsonl}"
  FAILED_OUTPUT_PATH="${FAILED_OUTPUT_PATH:-${PIPELINE_DIR}/generated.failed.jsonl}"
  VALIDATED_OUTPUT_PATH="${VALIDATED_OUTPUT_PATH:-${PIPELINE_DIR}/validated.jsonl}"
  VALIDATION_REPORTS_PATH="${VALIDATION_REPORTS_PATH:-${PIPELINE_DIR}/validation_reports.jsonl}"
  VALIDATION_FAILED_PATH="${VALIDATION_FAILED_PATH:-${PIPELINE_DIR}/validation.failed.jsonl}"
  REPAIR_HISTORY_PATH="${REPAIR_HISTORY_PATH:-${PIPELINE_DIR}/repair_history.jsonl}"
  TRAIN_OUTPUT_PATH="${TRAIN_OUTPUT_PATH:-${PIPELINE_DIR}/train.jsonl}"
  TRAIN_SUMMARY_PATH="${TRAIN_SUMMARY_PATH:-${PIPELINE_DIR}/train.summary.json}"

  export OUTPUT_DIR INPUT_PATH KB_DIR ANALYSIS_DIR PLANNING_DIR PIPELINE_DIR
  export KB_RECORDS_PATH KB_ENTITIES_PATH VICTIM_ANSWER_PATH VICTIM_ANSWER_RAW_PATH
  export STEP_EVALUATION_PATH MASTERY_PATH MASTERY_JSON_PATH PLAN_PATH PLAN_SUMMARY_PATH
  export GENERATED_OUTPUT_PATH RAW_OUTPUT_PATH FAILED_OUTPUT_PATH
  export VALIDATED_OUTPUT_PATH VALIDATION_REPORTS_PATH VALIDATION_FAILED_PATH REPAIR_HISTORY_PATH
  export TRAIN_OUTPUT_PATH TRAIN_SUMMARY_PATH

  mkdir -p "${KB_DIR}" "${ANALYSIS_DIR}" "${PLANNING_DIR}" "${PIPELINE_DIR}"
  STAGE_REMAINING_ARGS=("$@")
}

stage_log() {
  echo "[stage] $*" >&2
}

stage_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|y|Y|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

stage_skip_if_complete() {
  local label="$1"
  shift
  if stage_truthy "${STAGE_FORCE:-0}"; then
    return 1
  fi
  if ! stage_truthy "${STAGE_RESUME:-1}"; then
    return 1
  fi
  local path
  for path in "$@"; do
    if [[ ! -s "${path}" ]]; then
      return 1
    fi
  done
  stage_log "skip ${label}: existing outputs detected; set STAGE_FORCE=1 to rerun"
  for path in "$@"; do
    stage_log "  ${path}"
  done
  return 0
}

stage_normalize_model() {
  printf '%s' "${1:-}" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//; s:/*$::'
}

stage_model_basename() {
  local normalized
  normalized="$(stage_normalize_model "${1:-}")"
  normalized="${normalized//\\//}"
  printf '%s' "${normalized##*/}"
}

stage_models_match() {
  local served expected
  served="$(stage_normalize_model "${1:-}")"
  expected="$(stage_normalize_model "${2:-}")"
  [[ -n "${served}" && -n "${expected}" ]] || return 1
  [[ "${served}" == "${expected}" ]] && return 0
  [[ "$(stage_model_basename "${served}")" == "$(stage_model_basename "${expected}")" ]]
}

stage_probe_models() {
  "${PYTHON_BIN}" - <<'PY'
import json
import os
import sys
import urllib.error
import urllib.request

base_url = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8911/v1").rstrip("/")
api_key = os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY"
request = urllib.request.Request(
    base_url + "/models",
    headers={"Authorization": f"Bearer {api_key}"},
    method="GET",
)
try:
    with urllib.request.urlopen(request, timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
except urllib.error.HTTPError as exc:
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:
        body = ""
    print(f"ERROR HTTP {exc.code}: {body}", file=sys.stderr)
    raise SystemExit(1)
except Exception as exc:
    print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
    raise SystemExit(1)

models = []
if isinstance(payload, dict):
    data = payload.get("data")
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                model = item.get("id") or item.get("name")
                if model:
                    models.append(str(model).strip())
    else:
        model = payload.get("id") or payload.get("model") or payload.get("name")
        if model:
            models.append(str(model).strip())

for model in models:
    if model:
        print(model)
PY
}

stage_check_served_model() {
  local expected="$1"
  local models output status line
  output="$(stage_probe_models 2>&1)" && status=0 || status=$?
  if [[ "${status}" -ne 0 ]]; then
    stage_log "vLLM probe failed for expected model: ${expected}"
    stage_log "${output}"
    return 1
  fi
  models="${output}"
  while IFS= read -r line; do
    if stage_models_match "${line}" "${expected}"; then
      stage_log "vLLM external service ready: expected=${expected} served=${line}"
      return 0
    fi
  done <<< "${models}"
  stage_log "vLLM model mismatch: expected=${expected}"
  stage_log "served models:"
  if [[ -n "${models}" ]]; then
    printf '%s\n' "${models}" >&2
  else
    stage_log "<none>"
  fi
  return 1
}

stage_ensure_vllm() {
  local expected="$1"
  local label="${2:-model}"
  if [[ -z "${expected}" ]]; then
    stage_log "skip vLLM check for ${label}: empty model"
    return 0
  fi

  local mode="${STAGE_VLLM_MODE:-external}"
  local timeout="${STAGE_VLLM_WAIT_TIMEOUT:-0}"
  local poll="${STAGE_VLLM_POLL_SEC:-${VLLM_EXTERNAL_POLL_SEC:-5}}"
  stage_log "stage vLLM mode=${mode} label=${label} expected=${expected}"

  if [[ "${mode}" == "skip" ]]; then
    return 0
  fi

  if [[ "${mode}" == "managed" ]]; then
    local pid_file="${VLLM_PID_FILE:-${OUTPUT_DIR}/runtime/vllm/vllm.pid}"
    local log_file="${VLLM_LOG_FILE:-${OUTPUT_DIR}/runtime/vllm/vllm.log}"
    stage_log "starting managed vLLM for ${label}: ${expected}"
    "${STAGE_ROOT_DIR}/run/start_vllm.sh" \
      --background \
      --pid-file "${pid_file}" \
      --log-file "${log_file}" \
      --model "${expected}" >/dev/null
    if [[ "${STAGE_VLLM_STOP_ON_EXIT:-0}" == "1" ]]; then
      STAGE_MANAGED_PID_FILE="${pid_file}"
      export STAGE_MANAGED_PID_FILE
      trap stage_cleanup_managed_vllm EXIT
    fi
  elif [[ "${mode}" != "external" ]]; then
    echo "[stage] unsupported STAGE_VLLM_MODE=${mode}; use external, managed, or skip" >&2
    return 1
  fi

  local waited=0
  while true; do
    if stage_check_served_model "${expected}"; then
      return 0
    fi
    if [[ "${timeout}" -ge 0 && "${waited}" -ge "${timeout}" ]]; then
      stage_log "vLLM is not ready for ${label}; start the expected model externally or set STAGE_VLLM_MODE=managed"
      return 1
    fi
    stage_log "waiting for external vLLM ${label} (${waited}s/${timeout}s)"
    sleep "${poll}"
    waited=$((waited + poll))
  done
}

stage_require_file() {
  local path="$1"
  local hint="${2:-}"
  if [[ ! -f "${path}" ]]; then
    echo "[stage] required file missing: ${path}" >&2
    if [[ -n "${hint}" ]]; then
      echo "[stage] hint: ${hint}" >&2
    fi
    return 1
  fi
}

stage_cleanup_managed_vllm() {
  if [[ -n "${STAGE_MANAGED_PID_FILE:-}" ]]; then
    "${STAGE_ROOT_DIR}/run/stop_vllm.sh" --pid-file "${STAGE_MANAGED_PID_FILE}" >/dev/null 2>&1 || true
  fi
}
