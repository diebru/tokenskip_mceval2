#!/usr/bin/env bash
# Full explanation baseline (stage 1): every McEval language, ~2066 tasks.
# Stage 2 (round-trip code generation from the docstring) is run separately
# once these outputs exist.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Coder-7B-Instruct}"
OUT_DIR="${OUT_DIR:-../outputs/Qwen2.5-Coder-7B-Instruct/baseline/explanation}"
DATA_ROOT="${DATA_ROOT:-../McEval}"

python infer_mceval.py \
    --task explanation \
    --model-path "$MODEL_PATH" \
    --data-root "$DATA_ROOT" \
    --out-dir "$OUT_DIR" \
    --languages all \
    --max-tokens 1024 \
    --temperature 0.0 \
    --tensor-parallel-size 2
