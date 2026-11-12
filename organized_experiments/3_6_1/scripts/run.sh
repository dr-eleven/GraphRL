#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/temptest/.venvs/graphrl-3_5_1-3_6/bin/python}"
exec "$PYTHON_BIN" "$PROJECT/scripts/run_pipeline.py" "$@"
