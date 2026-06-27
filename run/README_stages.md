# Stage Scripts

These scripts split the full pipeline into independent stages. By default they
use an externally started vLLM server and do not start, switch, or stop vLLM.

```bash
cd /path/to/data_gradual_new
export STAGE_VLLM_MODE=external
```

Each script loads `config/pipeline.env`, resolves `PIPELINE_PYTHON`, and writes
outputs under:

```text
${OUTPUT_DIR}/kb/${DATASET_NAME}
${OUTPUT_DIR}/analysis/${DATASET_NAME}
${OUTPUT_DIR}/planning/${DATASET_NAME}
${OUTPUT_DIR}/pipeline/${DATASET_NAME}
```

The vLLM check calls `/v1/models` with `Authorization: Bearer ${VLLM_API_KEY}`.
It accepts full paths, trailing slashes, and basename-only model IDs as matches.

## Full Pipeline Compatibility

The legacy entrypoint is still supported:

```bash
bash run/run_full_pipeline.sh gsm8k
```

Unlike an individual numbered stage, the full pipeline wrapper defaults to
managed vLLM mode when `STAGE_VLLM_MODE` is not set. It starts or switches the
served model before each model-dependent stage and stops the managed service
when the sequence exits.

To use a manually started vLLM service instead:

```bash
STAGE_VLLM_MODE=external bash run/run_full_pipeline.sh gsm8k
```

In external mode you are responsible for switching the served model between
stages such as victim answering and Qwen-based generation/validation.

## Resume Policy

Resume is enabled by default:

```bash
STAGE_RESUME=1
```

Use this to force a stage to rebuild/regenerate from scratch:

```bash
STAGE_FORCE=1 bash run/05_generate_questions.sh gsm8k
```

Stage behavior:

| Stage | Recovery behavior |
| --- | --- |
| `01_build_kb.sh` | Skips if KB records and entities already exist. |
| `02_answer_seed.sh` | Resumes from existing `victim_answers.raw.jsonl`; saves every `ANSWER_CHECKPOINT_EVERY` answers. |
| `03_score_seed.sh` | Resumes from `step_evaluations.jsonl.partial`; completed records are appended as scoring finishes. |
| `04_build_synthesis_plan.sh` | Skips if plan and summary already exist. |
| `05_generate_questions.sh` | Resumes from existing `generated.jsonl`; skips successful `plan_id`s; saves every `GEN_CHECKPOINT_EVERY` completions. |
| `06_validate_generated.sh` | Saves canonical validation files after each validation round; skips if validated outputs already exist. |
| `07_export_training_data.sh` | Skips if train output and summary already exist. |

Responsibility split:

- `04_build_synthesis_plan.sh` is the diversity/similarity-control stage. It chooses knowledge focus, scene, problem pattern, target difficulty, and number strategy.
- `05_generate_questions.sh` only follows the plan and emits parseable `question`, `steps`, and numeric `answer`; it does not perform global similarity filtering.
- `06_validate_generated.sh` performs correctness, solvability, uniqueness, difficulty, repair, regeneration, and replan after repeated validation failures.

Useful checkpoint knobs:

```bash
ANSWER_RESUME=1
ANSWER_CHECKPOINT_EVERY=50
SCORE_RESUME=1
GEN_RESUME=1
GEN_CHECKPOINT_EVERY=50
```

Disable resume for long generation only when intentionally regenerating:

```bash
GEN_RESUME=0 bash run/05_generate_questions.sh gsm8k --no-resume
```

## Stages

1. Build KB, no vLLM required:

```bash
bash run/01_build_kb.sh gsm8k
```

2. Answer seed questions, requires `VICTIM_MODEL` already served:

```bash
bash run/02_answer_seed.sh gsm8k
```

3. Score seed answers and build mastery, requires `STEP_MODEL` already served:

```bash
bash run/03_score_seed.sh gsm8k
```

4. Build synthesis plan, no vLLM required:

```bash
bash run/04_build_synthesis_plan.sh gsm8k
```

5. Generate questions, requires `GEN_MODEL` already served:

```bash
bash run/05_generate_questions.sh gsm8k
```

6. Validate generated questions, requires `QC_MODEL` already served:

```bash
bash run/06_validate_generated.sh gsm8k
```

7. Export training data, no vLLM required:

```bash
bash run/07_export_training_data.sh gsm8k
```

## Optional Managed Mode

If you explicitly want a stage script to start vLLM:

```bash
STAGE_VLLM_MODE=managed bash run/05_generate_questions.sh gsm8k
```

To stop the managed vLLM when the stage exits:

```bash
STAGE_VLLM_MODE=managed STAGE_VLLM_STOP_ON_EXIT=1 bash run/05_generate_questions.sh gsm8k
```

## Common Overrides

```bash
DATASET_NAME=gsm8k
INPUT_PATH=/path/to/gsm8k.jsonl
OUTPUT_DIR=/path/to/outputs
N_ANSWERS=10
STAGE_VLLM_WAIT_TIMEOUT=0
```
