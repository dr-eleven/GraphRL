#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_PY="${BASE_PYTHON:-python3.10}"
ENV_DIR="${EXPERIMENT_ENV:-$ROOT/.venv}"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then "$BASE_PY" -m venv "$ENV_DIR"; fi
PY="$ENV_DIR/bin/python"
"$PY" -m pip install --upgrade pip setuptools wheel packaging ninja
"$PY" -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
"$PY" -m pip install -r "$ROOT/requirements-native.txt"
"$PY" -m pip install flash-attn==2.7.4.post1 --no-build-isolation
mkdir -p "$ROOT/verification"
"$PY" -m pip check
"$PY" -m pip freeze > "$ROOT/verification/environment_installed.txt"
echo "Environment installed. Download the model, then run code/preflight.py and the tests as documented in README.md."
