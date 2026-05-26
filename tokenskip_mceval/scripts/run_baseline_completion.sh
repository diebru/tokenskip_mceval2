#!/usr/bin/env bash
# Full completion baseline: every McEval language, ~10128 tasks total.
# At ~1800 tok/s with tp=2 expect roughly 2-3 hours wall clock.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Coder-7B-Instruct}"
OUT_DIR="${OUT_DIR:-../outputs/Qwen2.5-Coder-7B-Instruct/baseline/completion}"
DATA_ROOT="${DATA_ROOT:-../McEval}"

python infer_mceval.py \
    --task completion \
    --model-path "$MODEL_PATH" \
    --data-root "$DATA_ROOT" \
    --out-dir "$OUT_DIR" \
    --languages all \
    --max-tokens 1536 \
    --temperature 0.0 \
    --tensor-parallel-size 2
