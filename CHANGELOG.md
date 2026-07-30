# Changelog

## V1.1.1

- Reorganized configuration examples into complete machine-specific folders:
  `config/remote/` and `config/jizhicfs/`.
- Added ready-to-run GSM8K 10-answer and 20-answer legacy configs for both
  machines.
- Added complete `evaluate.env` examples for both machines.
- Expanded all jizhicfs model and project paths to concrete versioned paths
  under `/jizhicfs/hymiezhao/lpc/repos/brj/data_gradual_V1.1.1`.

## V1.1.0

- Fixed managed vLLM startup when shell scripts are not executable by invoking
  launcher scripts through `bash`.
- Made managed vLLM shutdown scoped to the configured `VLLM_PID_FILE` and
  `VLLM_API_PORT`; the stop script no longer scans and terminates all vLLM API
  servers on the machine.
- Reused an already running matching vLLM service instead of restarting it.
- Kept vLLM alive across full-pipeline stages and cleaned up only once when
  the sequence exits or is interrupted.
- Added machine-specific pipeline config examples and legacy GSM8K experiment
  overlays for 10-answer and 20-answer runs.

## V1.0.0

- Known-good baseline restored from commit
  `c3c694e2ad3bc9d4a7a7451155fc88c2d968c4c4`.
