#!/usr/bin/env bash

load_pipeline_config() {
  local root_dir="$1"
  local config_file="${root_dir}/config/pipeline.env"
  local example_file="${root_dir}/config/pipeline.example.env"
  local override_file="${PIPELINE_CONFIG_FILE:-}"
  if [[ -n "${override_file}" && "${override_file}" != /* ]]; then
    override_file="${root_dir}/${override_file}"
  fi

  set -a
  if [[ -f "${config_file}" ]]; then
    # shellcheck disable=SC1090
    source "${config_file}"
  elif [[ -f "${example_file}" ]]; then
    # shellcheck disable=SC1090
    source "${example_file}"
  fi
  if [[ -n "${override_file}" ]]; then
    if [[ ! -f "${override_file}" ]]; then
      echo "[env] PIPELINE_CONFIG_FILE not found: ${override_file}" >&2
      set +a
      return 1
    fi
    # shellcheck disable=SC1090
    source "${override_file}"
    export PIPELINE_CONFIG_FILE="${override_file}"
  fi
  export PIPELINE_CONFIG_LOADED=1
  set +a
}

resolve_conda_sh() {
  if [[ -n "${CONDA_SH:-}" && -f "${CONDA_SH}" ]]; then
    printf '%s\n' "${CONDA_SH}"
    return 0
  fi
  if command -v conda >/dev/null 2>&1; then
    local conda_base
    conda_base="$(conda info --base)"
    if [[ -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
      printf '%s\n' "${conda_base}/etc/profile.d/conda.sh"
      return 0
    fi
  fi
  return 1
}

activate_pipeline_env() {
  if [[ -n "${PIPELINE_PYTHON:-${BRJ_PYTHON:-}}" ]]; then
    echo "[env] pipeline Python configured explicitly; skipping conda activation" >&2
    return 0
  fi

  local env_name="${PIPELINE_CONDA_ENV:-brj}"
  if [[ -z "${env_name}" ]]; then
    return 0
  fi

  local conda_sh
  if ! conda_sh="$(resolve_conda_sh)"; then
    echo "[env] cannot locate conda.sh; set CONDA_SH in config/pipeline.env" >&2
    return 1
  fi
  # shellcheck disable=SC1090
  source "${conda_sh}"
  conda activate "${env_name}"
  echo "[env] pipeline conda environment: ${env_name}" >&2
}

resolve_pipeline_python() {
  local configured="${PIPELINE_PYTHON:-${BRJ_PYTHON:-}}"
  if [[ -n "${configured}" ]]; then
    if [[ ! -x "${configured}" ]]; then
      echo "[env] configured pipeline Python is not executable: ${configured}" >&2
      return 1
    fi
    printf '%s\n' "${configured}"
    return 0
  fi

  if ! command -v python >/dev/null 2>&1; then
    echo "[env] python not found after activating the pipeline environment" >&2
    return 1
  fi
  command -v python
}
