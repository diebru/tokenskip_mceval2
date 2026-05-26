#!/usr/bin/env bash
# Pilot: 10 Python tasks with Qwen2.5-Coder-7B-Instruct.
# Verifies CoT is being produced and that split_cot_code does its job.
# Run from the tokenskip_mceval/ folder on the server.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Coder-7B-Instruct}"
OUT_DIR="${OUT_DIR:-../outputs/Qwen2.5-Coder-7B-Instruct/baseline-pilot}"
DATA_DIR="${DATA_DIR:-../McEval/data}"

python infer_mceval.py \
    --model-path "$MODEL_PATH" \
    --data-dir "$DATA_DIR" \
    --out-dir "$OUT_DIR" \
    --languages Python \
    --limit 10 \
    --max-tokens 1536 \
    --temperature 0.0 \
    --tensor-parallel-size 1
