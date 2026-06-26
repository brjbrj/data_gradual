# Stage Scripts

These scripts split the full pipeline into independent stages. By default they
use an externally started vLLM server and do not start, switch, or stop vLLM.

## Defaults

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

## Managed Mode Optional

If you explicitly want a stage script to start vLLM:

```bash
STAGE_VLLM_MODE=managed bash run/05_generate_questions.sh gsm8k
```

To stop the managed vLLM when the stage exits:

```bash
STAGE_VLLM_MODE=managed STAGE_VLLM_STOP_ON_EXIT=1 bash run/05_generate_questions.sh gsm8k
```

## Useful Overrides

```bash
DATASET_NAME=gsm8k
INPUT_PATH=/path/to/gsm8k.jsonl
OUTPUT_DIR=/path/to/outputs
N_ANSWERS=10
STAGE_VLLM_WAIT_TIMEOUT=0
```
