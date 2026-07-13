#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper for older notes/commands that used stage number 07 for
# export before step refinement was inserted as the new Stage 07.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "[stage] 07_export_training_data.sh is kept for compatibility; forwarding to 08_export_training_data.sh" >&2
exec bash "${ROOT_DIR}/run/08_export_training_data.sh" "$@"
