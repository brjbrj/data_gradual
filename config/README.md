# Configuration Examples

Use one machine base file plus one experiment overlay.

```bash
cp config/machines/remote_root_base.env config/pipeline.env
PIPELINE_CONFIG_FILE=config/experiments/gsm8k_answer10_legacy_10_50_26x.env \
  bash run/run_full_pipeline.sh gsm8k
```

For the other machine:

```bash
cp config/machines/jizhicfs_base.env config/pipeline.env
PIPELINE_CONFIG_FILE=config/experiments/gsm8k_answer20_legacy_10_50_26x.env \
  bash run/run_full_pipeline.sh gsm8k
```

`V1.1.0` keeps the `V1.0.0` allocation algorithm. The runnable experiment
overlays included here are:

- `gsm8k_answer10_legacy_10_50_26x.env`: 10 victim answers, GSM8K, 26x synthesis, 10 to 50 generated questions per seed, validation/refinement rounds set to 50.
- `gsm8k_answer20_legacy_10_50_26x.env`: same allocation, but 20 victim answers per seed.

The threshold/dynamic allocation experiments are intentionally not included in
`V1.1.0` because that mechanism is not part of the restored baseline.

vLLM managed mode is isolated by `VLLM_API_PORT` and `VLLM_PID_FILE`. Use
different values for simultaneous experiments in different project folders.
Single numbered stages stop only the vLLM server they started. Full pipeline
runs keep the server alive between adjacent stages and clean it up once when
the sequence exits or receives Ctrl+C.
