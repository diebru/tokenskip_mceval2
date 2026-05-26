#!/usr/bin/env bash
# Overnight chain runner for the baseline phase.
#
# Each step writes its output to disk before the next starts, so if anything
# crashes you keep the earlier results. Each Python invocation exits before
# the next so vLLM never holds the GPU across steps.
#
# Launch with:   nohup bash scripts/run_overnight.sh > overnight.log 2>&1 &
# Tail with:     tail -f overnight.log

set -uo pipefail   # NOT -e: we want to continue past failures
cd "$(dirname "$0")/.."  # run from tokenskip_mceval/

DATA_ROOT="../McEval"
OUT_ROOT="../outputs"

# Each model gets its own subdir; baselines for the 3 typologies live under
# {OUT_ROOT}/{model_short}/baseline/{typology}
MODELS=(
    "Qwen/Qwen2.5-Coder-7B-Instruct|Qwen2.5-Coder-7B-Instruct|2"
    "Qwen/Qwen2.5-Coder-1.5B-Instruct|Qwen2.5-Coder-1.5B-Instruct|1"
    "Qwen/Qwen2.5-Coder-3B-Instruct|Qwen2.5-Coder-3B-Instruct|1"
)

log() { echo -e "\n=== [$(date '+%H:%M:%S')] $* ==="; }

for entry in "${MODELS[@]}"; do
    IFS="|" read -r MODEL_PATH MODEL_SHORT TP <<< "$entry"
    BASE="${OUT_ROOT}/${MODEL_SHORT}/baseline"
    EVAL="${OUT_ROOT}/${MODEL_SHORT}/eval-full"

    # --- Generation ---
    if [[ ! -f "${BASE}/generation/Python.jsonl" ]]; then
        log "${MODEL_SHORT}: generation inference"
        python infer_mceval.py \
            --task generation \
            --model-path "$MODEL_PATH" \
            --data-root "$DATA_ROOT" \
            --out-dir "${BASE}/generation" \
            --languages all \
            --max-tokens 1536 --temperature 0.0 \
            --tensor-parallel-size "$TP"
    else
        log "${MODEL_SHORT}: generation already done, skipping"
    fi

    log "${MODEL_SHORT}: generation Docker eval"
    IMAGE=mceval-full \
        RESULTS_DIR="${BASE}/generation" \
        SAVE_DIR="${EVAL}/generation" \
        bash scripts/run_eval_docker.sh

    # --- Completion ---
    log "${MODEL_SHORT}: completion inference (~2-3 h)"
    python infer_mceval.py \
        --task completion \
        --model-path "$MODEL_PATH" \
        --data-root "$DATA_ROOT" \
        --out-dir "${BASE}/completion" \
        --languages all \
        --max-tokens 1536 --temperature 0.0 \
        --tensor-parallel-size "$TP"

    log "${MODEL_SHORT}: completion Docker eval"
    IMAGE=mceval-full \
        RESULTS_DIR="${BASE}/completion" \
        SAVE_DIR="${EVAL}/completion" \
        bash scripts/run_eval_docker.sh

    # --- Explanation stage 1 (code -> docstring) ---
    log "${MODEL_SHORT}: explanation stage-1 inference"
    python infer_mceval.py \
        --task explanation \
        --model-path "$MODEL_PATH" \
        --data-root "$DATA_ROOT" \
        --out-dir "${BASE}/explanation" \
        --languages all \
        --max-tokens 1024 --temperature 0.0 \
        --tensor-parallel-size "$TP"

    # --- Explanation stage 2 (docstring -> code via round-trip) ---
    log "${MODEL_SHORT}: explanation stage-2 round-trip"
    python run_stage2_explanation.py \
        --model-path "$MODEL_PATH" \
        --stage1-dir "${BASE}/explanation" \
        --out-dir "${BASE}/explanation_stage2" \
        --languages all \
        --max-tokens 1536 --temperature 0.0 \
        --tensor-parallel-size "$TP"

    log "${MODEL_SHORT}: explanation Docker eval (on stage-2 code)"
    IMAGE=mceval-full \
        RESULTS_DIR="${BASE}/explanation_stage2" \
        SAVE_DIR="${EVAL}/explanation" \
        bash scripts/run_eval_docker.sh

    log "${MODEL_SHORT}: DONE"
done

log "All models complete."
