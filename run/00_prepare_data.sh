#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/run/stage_common.sh"
stage_init "$@"

if [[ -z "${STAGE_VLLM_MODE+x}" && -n "${VLLM_RUNTIME_MODE:-}" ]]; then
  export STAGE_VLLM_MODE="${VLLM_RUNTIME_MODE}"
fi

FORMAT_TEMPLATE="${DATA_FORMAT_TEMPLATE:-gsm8k}"
RAW_DATA_PATH="${RAW_INPUT_PATH:-${INPUT_PATH}}"

ARGS=("${PYTHON_BIN}" "${ROOT_DIR}/run/prepare_data.py"
  --input "${RAW_DATA_PATH}"
  --output "${PREPARED_INPUT_PATH}"
  --format-template "${FORMAT_TEMPLATE}"
  --classify-prompt "${CLASSIFY_PROMPT_PATH:-${ROOT_DIR}/prompt/classify.json}"
  --model "${CLASSIFY_MODEL:-${VLLM_MODEL:-}}"
  --base-url "${CLASSIFY_BASE_URL:-${VLLM_BASE_URL:-}}"
  --api-key "${CLASSIFY_API_KEY:-${VLLM_API_KEY:-EMPTY}}"
  --concurrency "${CLASSIFY_CONCURRENCY:-16}"
  --temperature "${CLASSIFY_TEMPERATURE:-0.1}"
  --top-p "${CLASSIFY_TOP_P:-0.9}"
  --max-tokens "${CLASSIFY_MAX_TOKENS:-50}"
  --classify-max-retries "${CLASSIFY_MAX_RETRIES:-3}"
)

if [[ -n "${SAMPLE_LIMIT:-}" ]]; then
  ARGS+=(--sample-limit "${SAMPLE_LIMIT}")
fi
if stage_truthy "${PREPARE_FORCE_FORMAT:-0}"; then
  ARGS+=(--force-format)
fi
if stage_truthy "${PREPARE_SKIP_FORMAT:-0}"; then
  ARGS+=(--skip-format)
fi
if ! stage_truthy "${PREPARE_CLASSIFY:-1}"; then
  ARGS+=(--no-classify)
fi
if stage_truthy "${CLASSIFY_OVERWRITE_EXISTING:-0}"; then
  ARGS+=(--overwrite-classification)
fi
ARGS+=("${STAGE_REMAINING_ARGS[@]}")

if stage_skip_if_complete "00_prepare_data" "${PREPARED_INPUT_PATH}"; then
  exit 0
fi

NEEDS_CLASSIFY=0
if stage_truthy "${PREPARE_CLASSIFY:-1}"; then
  if stage_truthy "${CLASSIFY_OVERWRITE_EXISTING:-0}"; then
    NEEDS_CLASSIFY=1
  else
    NEEDS_CLASSIFY="$("${PYTHON_BIN}" -c '
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
input_path = Path(sys.argv[2])
sys.path.insert(0, str(root))
from kb_pipeline.data_prepare import inspect_jsonl_schema

schema = inspect_jsonl_schema(input_path)
print(1 if schema["needs_classify"] else 0)
' "${ROOT_DIR}" "${RAW_DATA_PATH}")"
  fi
fi

if [[ "${NEEDS_CLASSIFY}" == "1" ]]; then
  stage_ensure_vllm "${CLASSIFY_MODEL:-${VLLM_MODEL:-}}" "question classification"
else
  stage_log "00 prepare_data classification vLLM check skipped: existing question_type detected or PREPARE_CLASSIFY=0"
fi

stage_log "00 prepare_data input=${RAW_DATA_PATH} output=${PREPARED_INPUT_PATH} format=${FORMAT_TEMPLATE} classify=${PREPARE_CLASSIFY:-1}"
"${ARGS[@]}"
