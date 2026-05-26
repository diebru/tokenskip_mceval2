#!/usr/bin/env bash
# Explanation pilot: 10 Python tasks. Verifies the docstring prompt and
# the <answer>...</answer> splitter.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Coder-7B-Instruct}"
OUT_DIR="${OUT_DIR:-../outputs/Qwen2.5-Coder-7B-Instruct/baseline-pilot/explanation}"
DATA_ROOT="${DATA_ROOT:-../McEval}"

python infer_mceval.py \
    --task explanation \
    --model-path "$MODEL_PATH" \
    --data-root "$DATA_ROOT" \
    --out-dir "$OUT_DIR" \
    --languages Python \
    --limit 10 \
    --max-tokens 1024 \
    --temperature 0.0 \
    --tensor-parallel-size 1
