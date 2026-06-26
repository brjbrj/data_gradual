#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/run/stage_common.sh"
stage_init "$@"

stage_require_file "${MASTERY_PATH}" "run: bash run/03_score_seed.sh ${DATASET_NAME}"
stage_require_file "${KB_RECORDS_PATH}" "run: bash run/01_build_kb.sh ${DATASET_NAME}"
stage_require_file "${KB_ENTITIES_PATH}" "run: bash run/01_build_kb.sh ${DATASET_NAME}"

stage_log "04 build_synthesis_plan output=${PLAN_PATH}"
"${PYTHON_BIN}" "${ROOT_DIR}/run/build_synthesis_plan.py" \
  --mastery "${MASTERY_PATH}" \
  --kb-records "${KB_RECORDS_PATH}" \
  --entities "${KB_ENTITIES_PATH}" \
  --output "${PLAN_PATH}" \
  --summary-output "${PLAN_SUMMARY_PATH}" \
  "${STAGE_REMAINING_ARGS[@]}"
