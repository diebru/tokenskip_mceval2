#!/usr/bin/env bash
# Build LLaMA-Factory SFT datasets for every (model, typology) we have data for.
# Joins compressed CoTs with eval-detail (correctness) and emits one JSON per
# dataset under outputs/{model}/train/{typology}.json.
set -uo pipefail
cd "$(dirname "$0")/.."

OUT_ROOT="../outputs"
RATIOS="${RATIOS:-0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0}"

JOBS=(
    "Qwen2.5-Coder-7B-Instruct|generation"
    "Qwen2.5-Coder-7B-Instruct|completion"
    "Qwen2.5-Coder-7B-Instruct|explanation"
    "Qwen2.5-Coder-3B-Instruct|generation"
    "Qwen2.5-Coder-3B-Instruct|completion"
    "Qwen2.5-Coder-1.5B-Instruct|generation"
    "Qwen2.5-Coder-1.5B-Instruct|completion"
)

log() { echo -e "\n=== [$(date '+%H:%M:%S')] $* ==="; }

for entry in "${JOBS[@]}"; do
    IFS="|" read -r MODEL TYP <<< "$entry"
    OUT="${OUT_ROOT}/${MODEL}/train/${TYP}.json"
    log "${MODEL}/${TYP} -> ${OUT}"
    python build_training_data.py \
        --model "$MODEL" \
        --typology "$TYP" \
        --ratios $RATIOS \
        --out "$OUT" || echo "[skip] ${MODEL}/${TYP} failed"
done

log "All training datasets built."
