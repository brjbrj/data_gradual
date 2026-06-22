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
MASTERY_PATH="${MASTERY_PATH:-${PIPELINE_OUTPUT_DIR}/analysis/${DATASET_NAME}/mastery_records.jsonl}"
KB_RECORDS_PATH="${KB_RECORDS_PATH:-${PIPELINE_OUTPUT_DIR}/kb/${DATASET_NAME}/records.jsonl}"
KB_ENTITIES_PATH="${KB_ENTITIES_PATH:-${PIPELINE_OUTPUT_DIR}/kb/${DATASET_NAME}/entities.json}"
PLAN_OUTPUT_PATH="${PLAN_OUTPUT_PATH:-${PIPELINE_OUTPUT_DIR}/planning/${DATASET_NAME}/synthesis_plan.jsonl}"

exec "${PYTHON_BIN}" "${ROOT_DIR}/run/build_synthesis_plan.py" \
  --mastery "${MASTERY_PATH}" \
  --kb-records "${KB_RECORDS_PATH}" \
  --entities "${KB_ENTITIES_PATH}" \
  --output "${PLAN_OUTPUT_PATH}" \
  "$@"
