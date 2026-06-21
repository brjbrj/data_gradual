#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${ROOT_DIR}/config/pipeline.env"

if [[ -f "${CONFIG_FILE}" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "${CONFIG_FILE}"
  set +a
fi

PYTHON_BIN="${BRJ_PYTHON:-/root/miniconda3/envs/brj/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON:-python}"
fi

DATASET_NAME="${DATASET_NAME:-gsm8k}"
if [[ $# -gt 0 && "${1:-}" != --* ]]; then
  DATASET_NAME="$1"
  shift
fi

PIPELINE_OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs}"
PLAN_PATH="${PLAN_PATH:-${PIPELINE_OUTPUT_DIR}/planning/${DATASET_NAME}/synthesis_plan.jsonl}"
MASTERY_PATH="${MASTERY_PATH:-${PIPELINE_OUTPUT_DIR}/analysis/${DATASET_NAME}/mastery_records.jsonl}"
GENERATED_OUTPUT_PATH="${GENERATED_OUTPUT_PATH:-${PIPELINE_OUTPUT_DIR}/pipeline/${DATASET_NAME}/generated.jsonl}"
RAW_OUTPUT_PATH="${RAW_OUTPUT_PATH:-${PIPELINE_OUTPUT_DIR}/pipeline/${DATASET_NAME}/generated.raw.jsonl}"
FAILED_OUTPUT_PATH="${FAILED_OUTPUT_PATH:-${PIPELINE_OUTPUT_DIR}/pipeline/${DATASET_NAME}/generated.failed.jsonl}"

exec "${PYTHON_BIN}" "${ROOT_DIR}/run/generate_questions.py" \
  --plan "${PLAN_PATH}" \
  --mastery "${MASTERY_PATH}" \
  --output "${GENERATED_OUTPUT_PATH}" \
  --raw-output "${RAW_OUTPUT_PATH}" \
  --failed-output "${FAILED_OUTPUT_PATH}" \
  "$@"
