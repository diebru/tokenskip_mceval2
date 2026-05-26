#!/usr/bin/env bash
# Full baseline: every McEval language with Qwen2.5-Coder-7B-Instruct.
# ~2007 tasks. With 2x A6000 use tensor-parallel-size=2 for headroom.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Coder-7B-Instruct}"
OUT_DIR="${OUT_DIR:-../outputs/Qwen2.5-Coder-7B-Instruct/baseline}"
DATA_DIR="${DATA_DIR:-../McEval/data}"

python infer_mceval.py \
    --model-path "$MODEL_PATH" \
    --data-dir "$DATA_DIR" \
    --out-dir "$OUT_DIR" \
    --languages all \
    --max-tokens 1536 \
    --temperature 0.0 \
    --tensor-parallel-size 2
