#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/run/stage_common.sh"

show_usage() {
  cat <<'USAGE'
Usage:
  bash ablations/run_ablation.sh <dataset> <variant> [--skip-generate] [--run-validation] [--run-refine] [--export]

Variants:
  answer_accuracy_only  Recompute mastery/allocation from final-answer accuracy only.
  hard_all              Preserve computed counts, force every target difficulty to Hard.
  equal_all             Preserve computed counts, force every target difficulty to Equal.
  easy_all              Preserve computed counts, force every target difficulty to Easy.
  uniform_count         Preserve computed difficulty, force every seed to the same count.
  identity              Copy original mastery unchanged into an ablation directory.

Useful environment overrides:
  ABLATION_ROOT_DIR          Default: ${OUTPUT_DIR}/ablations
  ABLATION_UNIFORM_COUNT     Optional explicit count for uniform_count.
  STAGE_FORCE=1             Rebuild existing ablation outputs.
  STAGE_VLLM_MODE=managed   Auto-start/switch vLLM, same as the main stage sequence.

Prerequisite:
  Run the main stages through 03 first so KB records, victim answers, and
  original mastery records exist. This script does not modify those files.
USAGE
}

if [[ $# -lt 2 ]]; then
  show_usage >&2
  exit 2
fi

DATASET_ARG="$1"
VARIANT="$2"
shift 2

RUN_GENERATE=1
RUN_VALIDATION_ABLATION=0
RUN_REFINE_ABLATION=0
RUN_EXPORT_ABLATION=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-generate)
      RUN_GENERATE=0
      ;;
    --run-validation)
      RUN_VALIDATION_ABLATION=1
      ;;
    --run-refine)
      RUN_VALIDATION_ABLATION=1
      RUN_REFINE_ABLATION=1
      ;;
    --export)
      RUN_EXPORT_ABLATION=1
      ;;
    --help|-h)
      show_usage
      exit 0
      ;;
    *)
      echo "[ablation] unknown argument: $1" >&2
      show_usage >&2
      exit 2
      ;;
  esac
  shift
done

stage_init "${DATASET_ARG}"

case "${VARIANT}" in
  answer_accuracy_only|hard_all|equal_all|easy_all|uniform_count|identity) ;;
  *)
    echo "[ablation] unsupported variant: ${VARIANT}" >&2
    show_usage >&2
    exit 2
    ;;
esac

ABLATION_ROOT_DIR="${ABLATION_ROOT_DIR:-${OUTPUT_DIR}/ablations}"
ABLATION_DIR="${ABLATION_ROOT_DIR}/${DATASET_NAME}/${VARIANT}"
mkdir -p "${ABLATION_DIR}"

ABLATION_MASTERY_PATH="${ABLATION_MASTERY_PATH:-${ABLATION_DIR}/mastery_records.jsonl}"
ABLATION_MASTERY_SUMMARY_PATH="${ABLATION_MASTERY_SUMMARY_PATH:-${ABLATION_DIR}/mastery_records.summary.json}"
ABLATION_PLAN_PATH="${ABLATION_PLAN_PATH:-${ABLATION_DIR}/synthesis_plan.jsonl}"
ABLATION_PLAN_SUMMARY_PATH="${ABLATION_PLAN_SUMMARY_PATH:-${ABLATION_DIR}/synthesis_plan.summary.json}"
ABLATION_GENERATED_OUTPUT_PATH="${ABLATION_GENERATED_OUTPUT_PATH:-${ABLATION_DIR}/generated.jsonl}"
ABLATION_RAW_OUTPUT_PATH="${ABLATION_RAW_OUTPUT_PATH:-${ABLATION_DIR}/generated.raw.jsonl}"
ABLATION_FAILED_OUTPUT_PATH="${ABLATION_FAILED_OUTPUT_PATH:-${ABLATION_DIR}/generated.failed.jsonl}"
ABLATION_VALIDATED_OUTPUT_PATH="${ABLATION_VALIDATED_OUTPUT_PATH:-${ABLATION_DIR}/validated.jsonl}"
ABLATION_VALIDATION_REPORTS_PATH="${ABLATION_VALIDATION_REPORTS_PATH:-${ABLATION_DIR}/validation_reports.jsonl}"
ABLATION_VALIDATION_FAILED_PATH="${ABLATION_VALIDATION_FAILED_PATH:-${ABLATION_DIR}/validation.failed.jsonl}"
ABLATION_REPAIR_HISTORY_PATH="${ABLATION_REPAIR_HISTORY_PATH:-${ABLATION_DIR}/repair_history.jsonl}"
ABLATION_REFINED_OUTPUT_PATH="${ABLATION_REFINED_OUTPUT_PATH:-${ABLATION_DIR}/refined.jsonl}"
ABLATION_REFINE_RAW_OUTPUT_PATH="${ABLATION_REFINE_RAW_OUTPUT_PATH:-${ABLATION_DIR}/refine.raw.jsonl}"
ABLATION_REFINE_FAILED_PATH="${ABLATION_REFINE_FAILED_PATH:-${ABLATION_DIR}/refine.failed.jsonl}"
ABLATION_REFINE_SUMMARY_PATH="${ABLATION_REFINE_SUMMARY_PATH:-${ABLATION_DIR}/refine.summary.json}"
ABLATION_TRAIN_OUTPUT_PATH="${ABLATION_TRAIN_OUTPUT_PATH:-${ABLATION_DIR}/train.jsonl}"
ABLATION_TRAIN_SUMMARY_PATH="${ABLATION_TRAIN_SUMMARY_PATH:-${ABLATION_DIR}/train.summary.json}"

stage_require_file "${KB_RECORDS_PATH}" "run: bash run/01_build_kb.sh ${DATASET_NAME}"
stage_require_file "${KB_ENTITIES_PATH}" "run: bash run/01_build_kb.sh ${DATASET_NAME}"

if [[ "${VARIANT}" == "answer_accuracy_only" ]]; then
  stage_require_file "${VICTIM_ANSWER_RAW_PATH}" "run: bash run/02_answer_seed.sh ${DATASET_NAME}"
  VARIANT_ARGS=(--answers "${VICTIM_ANSWER_RAW_PATH}")
else
  stage_require_file "${MASTERY_PATH}" "run: bash run/03_score_seed.sh ${DATASET_NAME}"
  VARIANT_ARGS=(--mastery "${MASTERY_PATH}")
fi

if [[ "${VARIANT}" == "uniform_count" && -n "${ABLATION_UNIFORM_COUNT:-}" ]]; then
  VARIANT_ARGS+=(--uniform-count "${ABLATION_UNIFORM_COUNT}")
fi

if [[ ! -s "${ABLATION_MASTERY_PATH}" || "${STAGE_FORCE:-0}" == "1" ]]; then
  stage_log "ablation ${VARIANT}: build mastery variant=${ABLATION_MASTERY_PATH}"
  "${PYTHON_BIN}" "${ROOT_DIR}/ablations/build_mastery_variant.py" \
    --variant "${VARIANT}" \
    --seed-input "${KB_RECORDS_PATH}" \
    --output "${ABLATION_MASTERY_PATH}" \
    --summary-output "${ABLATION_MASTERY_SUMMARY_PATH}" \
    --synthesis-target-multiplier "${SYNTHESIS_TARGET_MULTIPLIER:-26}" \
    --synthesis-min-per-seed "${SYNTHESIS_MIN_PER_SEED:-10}" \
    --synthesis-max-per-seed "${SYNTHESIS_MAX_PER_SEED:-50}" \
    --synthesis-balance-lambda "${SYNTHESIS_BALANCE_LAMBDA:-0.3}" \
    "${VARIANT_ARGS[@]}"
else
  stage_log "ablation ${VARIANT}: reuse mastery variant=${ABLATION_MASTERY_PATH}"
fi

export MASTERY_PATH="${ABLATION_MASTERY_PATH}"
export PLAN_PATH="${ABLATION_PLAN_PATH}"
export PLAN_SUMMARY_PATH="${ABLATION_PLAN_SUMMARY_PATH}"
export GENERATED_OUTPUT_PATH="${ABLATION_GENERATED_OUTPUT_PATH}"
export RAW_OUTPUT_PATH="${ABLATION_RAW_OUTPUT_PATH}"
export FAILED_OUTPUT_PATH="${ABLATION_FAILED_OUTPUT_PATH}"
export VALIDATED_OUTPUT_PATH="${ABLATION_VALIDATED_OUTPUT_PATH}"
export VALIDATION_REPORTS_PATH="${ABLATION_VALIDATION_REPORTS_PATH}"
export VALIDATION_FAILED_PATH="${ABLATION_VALIDATION_FAILED_PATH}"
export REPAIR_HISTORY_PATH="${ABLATION_REPAIR_HISTORY_PATH}"
export REFINED_OUTPUT_PATH="${ABLATION_REFINED_OUTPUT_PATH}"
export REFINE_RAW_OUTPUT_PATH="${ABLATION_REFINE_RAW_OUTPUT_PATH}"
export REFINE_FAILED_PATH="${ABLATION_REFINE_FAILED_PATH}"
export REFINE_SUMMARY_PATH="${ABLATION_REFINE_SUMMARY_PATH}"
export TRAIN_OUTPUT_PATH="${ABLATION_TRAIN_OUTPUT_PATH}"
export TRAIN_SUMMARY_PATH="${ABLATION_TRAIN_SUMMARY_PATH}"

bash "${ROOT_DIR}/run/04_build_synthesis_plan.sh" "${DATASET_NAME}"

if [[ "${RUN_GENERATE}" == "1" ]]; then
  bash "${ROOT_DIR}/run/05_generate_questions.sh" "${DATASET_NAME}"
fi

if [[ "${RUN_VALIDATION_ABLATION}" == "1" ]]; then
  bash "${ROOT_DIR}/run/06_validate_generated.sh" "${DATASET_NAME}"
fi

if [[ "${RUN_REFINE_ABLATION}" == "1" ]]; then
  bash "${ROOT_DIR}/run/07_refine_solution_steps.sh" "${DATASET_NAME}"
fi

if [[ "${RUN_EXPORT_ABLATION}" == "1" ]]; then
  if [[ "${RUN_REFINE_ABLATION}" == "1" ]]; then
    EXPORT_INPUT_PATH="${ABLATION_REFINED_OUTPUT_PATH}" bash "${ROOT_DIR}/run/08_export_training_data.sh" "${DATASET_NAME}"
  elif [[ "${RUN_VALIDATION_ABLATION}" == "1" ]]; then
    EXPORT_INPUT_PATH="${ABLATION_VALIDATED_OUTPUT_PATH}" bash "${ROOT_DIR}/run/08_export_training_data.sh" "${DATASET_NAME}"
  else
    EXPORT_INPUT_PATH="${ABLATION_GENERATED_OUTPUT_PATH}" bash "${ROOT_DIR}/run/08_export_training_data.sh" "${DATASET_NAME}"
  fi
fi

cat <<EOF
{
  "variant": "${VARIANT}",
  "ablation_dir": "${ABLATION_DIR}",
  "mastery": "${ABLATION_MASTERY_PATH}",
  "plan": "${ABLATION_PLAN_PATH}",
  "generated": "${ABLATION_GENERATED_OUTPUT_PATH}",
  "validated": "${ABLATION_VALIDATED_OUTPUT_PATH}",
  "refined": "${ABLATION_REFINED_OUTPUT_PATH}",
  "train": "${ABLATION_TRAIN_OUTPUT_PATH}"
}
EOF
