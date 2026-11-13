#!/usr/bin/env bash
# Portable environment matching the verified Python 3.11 / torch cu124 stack.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GRAPH_ROOT="$(cd -- "$ROOT/../.." && pwd)"
ENV_DIR="${EXPERIMENT_ENV:-$GRAPH_ROOT/.venv-3_5_1_2}"
PYTHON="${BOOTSTRAP_PYTHON:-python3.11}"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,11), sys.version'
if [[ ! -x "$ENV_DIR/bin/python" ]]; then "$PYTHON" -m venv "$ENV_DIR"; fi
PY="$ENV_DIR/bin/python"
"$PY" -m pip install pip==26.2.2 setuptools==84.2.1 wheel==0.48.0 packaging==26.3 ninja==1.13.2
"$PY" -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
DS_BUILD_OPS=0 "$PY" -m pip install --no-build-isolation -r "$ROOT/requirement.txt"
mkdir -p "$ROOT/verification"
"$PY" -m pip freeze > "$ROOT/verification/restored_environment_freeze.txt"
"$PY" -c 'import sys,torch,trl,transformers,peft,deepspeed; print(sys.version); print("torch",torch.__version__,"CUDA",torch.version.cuda,"TRL",trl.__version__,"transformers",transformers.__version__,"peft",peft.__version__)'
printf 'export PYTHON_BIN=%q\n' "$PY"
