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
GENERATED_PATH="${GENERATED_PATH:-${PIPELINE_OUTPUT_DIR}/pipeline/${DATASET_NAME}/generated.jsonl}"
PLAN_PATH="${PLAN_PATH:-${PIPELINE_OUTPUT_DIR}/planning/${DATASET_NAME}/synthesis_plan.jsonl}"
MASTERY_PATH="${MASTERY_PATH:-${PIPELINE_OUTPUT_DIR}/analysis/${DATASET_NAME}/mastery_records.jsonl}"
VALIDATED_OUTPUT_PATH="${VALIDATED_OUTPUT_PATH:-${PIPELINE_OUTPUT_DIR}/pipeline/${DATASET_NAME}/validated.jsonl}"
VALIDATION_REPORTS_PATH="${VALIDATION_REPORTS_PATH:-${PIPELINE_OUTPUT_DIR}/pipeline/${DATASET_NAME}/validation_reports.jsonl}"
VALIDATION_FAILED_PATH="${VALIDATION_FAILED_PATH:-${PIPELINE_OUTPUT_DIR}/pipeline/${DATASET_NAME}/validation.failed.jsonl}"
REPAIR_HISTORY_PATH="${REPAIR_HISTORY_PATH:-${PIPELINE_OUTPUT_DIR}/pipeline/${DATASET_NAME}/repair_history.jsonl}"

exec "${PYTHON_BIN}" "${ROOT_DIR}/run/validate_generated.py" \
  --generated "${GENERATED_PATH}" \
  --plan "${PLAN_PATH}" \
  --mastery "${MASTERY_PATH}" \
  --output "${VALIDATED_OUTPUT_PATH}" \
  --reports-output "${VALIDATION_REPORTS_PATH}" \
  --failed-output "${VALIDATION_FAILED_PATH}" \
  --repair-history-output "${REPAIR_HISTORY_PATH}" \
  "$@"
