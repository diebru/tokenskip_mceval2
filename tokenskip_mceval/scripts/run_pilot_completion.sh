#!/usr/bin/env bash
# Completion pilot: 10 Python tasks. Verifies the [MASK]-filling prompt
# and that the fence splitter still works on completion outputs.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Coder-7B-Instruct}"
OUT_DIR="${OUT_DIR:-../outputs/Qwen2.5-Coder-7B-Instruct/baseline-pilot/completion}"
DATA_ROOT="${DATA_ROOT:-../McEval}"

python infer_mceval.py \
    --task completion \
    --model-path "$MODEL_PATH" \
    --data-root "$DATA_ROOT" \
    --out-dir "$OUT_DIR" \
    --languages Python \
    --limit 10 \
    --max-tokens 1536 \
    --temperature 0.0 \
    --tensor-parallel-size 1
