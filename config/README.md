# Configuration Examples

`V1.1.1` uses complete machine-and-experiment config files. Copy the file you
want directly to `config/pipeline.env`, then run the pipeline. You should not
need to fill in model or project paths for the provided machines.

Remote server examples:

```bash
cp config/remote/gsm8k_answer10_legacy_10_50_26x.env config/pipeline.env
bash run/run_full_pipeline.sh gsm8k

cp config/remote/gsm8k_answer20_legacy_10_50_26x.env config/pipeline.env
bash run/run_full_pipeline.sh gsm8k
```

jizhicfs examples:

```bash
cp config/jizhicfs/gsm8k_answer10_legacy_10_50_26x.env config/pipeline.env
bash run/run_full_pipeline.sh gsm8k

cp config/jizhicfs/gsm8k_answer20_legacy_10_50_26x.env config/pipeline.env
bash run/run_full_pipeline.sh gsm8k
```

Evaluation examples are also complete:

```bash
cp config/remote/evaluate.env config/evaluate.env
cp config/jizhicfs/evaluate.env config/evaluate.env
```

Included pipeline configs:

- `config/remote/gsm8k_answer10_legacy_10_50_26x.env`
- `config/remote/gsm8k_answer20_legacy_10_50_26x.env`
- `config/jizhicfs/gsm8k_answer10_legacy_10_50_26x.env`
- `config/jizhicfs/gsm8k_answer20_legacy_10_50_26x.env`

Both GSM8K configs use the restored `V1.0.0` allocation algorithm:

- 26x synthesis multiplier
- minimum 10 and maximum 50 generated questions per seed
- validation rounds set to 50
- step-refinement rounds set to 50

The threshold/dynamic allocation experiments are not included in `V1.1.1`
because that mechanism is not part of the restored baseline.

For simultaneous experiments on one machine, copy the project directory and
change all four of these values in that copy:

- `VLLM_BASE_URL`
- `VLLM_API_PORT`
- `VLLM_PID_FILE`
- `VLLM_CUDA_VISIBLE_DEVICES`

Managed vLLM shutdown is scoped by `VLLM_PID_FILE` and `VLLM_API_PORT`, so
different experiments do not stop each other's vLLM services when these values
are distinct.
