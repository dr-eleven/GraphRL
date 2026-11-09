#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON_BIN:-/opt/data/private/.conda/envs/GraphLLM/bin/python}"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT/code${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -m unittest discover -s "$ROOT/tests" -v
