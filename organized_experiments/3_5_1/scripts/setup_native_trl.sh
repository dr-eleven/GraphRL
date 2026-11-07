#!/usr/bin/env bash
# Stable releases verified on 2026-11-02; Python 3.11 is required by current TRL.
set -euo pipefail
export UV_LINK_MODE=copy
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/opt/data/private/.local/share/uv/python}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
UV="${UV_BIN:-/opt/data/private/.conda/envs/GraphLLM/bin/uv}"
ENV="${EXPERIMENT_ENV:-/opt/data/private/.venvs/graphrl-3_3-trl-latest}"
if [[ ! -x "$UV" ]]; then
    /opt/data/private/.conda/envs/GraphLLM/bin/python -m pip install uv
fi
"$UV" python install 3.11
if [[ ! -x "$ENV/bin/python" ]]; then "$UV" venv --allow-existing --python 3.11 "$ENV"; fi
"$UV" pip install --python "$ENV/bin/python" torch==2.6.0 trl==1.17.1 transformers==5.22.1 \
    peft==0.21.2 accelerate==1.15.0 datasets==5.1.0 pip==26.2.2 pytest setuptools wheel ninja
DS_BUILD_OPS=0 "$UV" pip install --python "$ENV/bin/python" --no-build-isolation deepspeed==0.19.9
# Match the A-7B kernels. This wheel targets Python 3.11, torch 2.6, CUDA 12,
# Linux x86_64 and the torch wheel's CXX11_ABI=False; no local CUDA build is needed.
"$UV" pip install --python "$ENV/bin/python" liger-kernel==0.8.0 einops==0.8.2
FLASH_WHEEL="${FLASH_ATTN_WHEEL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl#sha256=58853b28a5a926cae14402bfd8d4d93a45ebf8f9e79533f37ab09d0d77a99c05}"
"$UV" pip install --python "$ENV/bin/python" --no-deps "$FLASH_WHEEL"
"$UV" pip freeze --python "$ENV/bin/python" > "$ROOT/verification/native_environment_freeze.txt"
"$ENV/bin/python" -c 'import sys,trl,transformers; print(sys.version); print("TRL",trl.__version__,"Transformers",transformers.__version__)'
