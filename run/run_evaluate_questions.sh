#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/run/common_env.sh"
load_pipeline_config "${ROOT_DIR}"
activate_pipeline_env
PYTHON_BIN="$(resolve_pipeline_python)"

INPUT_PATH="${EVALUATION_INPUT_PATH:-${ROOT_DIR}/outputs/pipeline/gsm8k/generated.jsonl}"
SOURCE_MAP="${SOURCE_MAP:-${ROOT_DIR}/outputs/pipeline/gsm8k/source_map.json}"
TARGET_MAP="${TARGET_MAP:-${ROOT_DIR}/outputs/pipeline/gsm8k/target_map.json}"
OUTPUT_PATH="${OUTPUT_PATH:-${ROOT_DIR}/outputs/pipeline/gsm8k/evaluated.jsonl}"

exec "${PYTHON_BIN}" "${ROOT_DIR}/run/evaluate_questions.py" \
  --input "${INPUT_PATH}" \
  --source-map "${SOURCE_MAP}" \
  --target-map "${TARGET_MAP}" \
  --output "${OUTPUT_PATH}"
