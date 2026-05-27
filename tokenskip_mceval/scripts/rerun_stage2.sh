#!/usr/bin/env bash
# Re-run explanation stage-2 + eval for the models whose overnight run hit
# the KeyError('test') bug (7B and 1.5B). Stage-1 outputs are intact; only
# stage-2 needs to be redone.
set -uo pipefail
cd "$(dirname "$0")/.."

OUT_ROOT="../outputs"

# Wipe broken stage-2 dirs first so the eval doesn't see the empty AWK.jsonl
# from the previous crash and divide-by-zero again.
rm -rf "${OUT_ROOT}/Qwen2.5-Coder-7B-Instruct/baseline/explanation_stage2"
rm -rf "${OUT_ROOT}/Qwen2.5-Coder-1.5B-Instruct/baseline/explanation_stage2"
rm -f  "${OUT_ROOT}/Qwen2.5-Coder-7B-Instruct/eval-full/explanation/results.jsonl"
rm -f  "${OUT_ROOT}/Qwen2.5-Coder-1.5B-Instruct/eval-full/explanation/results.jsonl"

MODELS=(
    "Qwen/Qwen2.5-Coder-7B-Instruct|Qwen2.5-Coder-7B-Instruct|2"
    "Qwen/Qwen2.5-Coder-1.5B-Instruct|Qwen2.5-Coder-1.5B-Instruct|1"
)

log() { echo -e "\n=== [$(date '+%H:%M:%S')] $* ==="; }

for entry in "${MODELS[@]}"; do
    IFS="|" read -r MODEL_PATH MODEL_SHORT TP <<< "$entry"
    BASE="${OUT_ROOT}/${MODEL_SHORT}/baseline"
    EVAL="${OUT_ROOT}/${MODEL_SHORT}/eval-full"

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

    log "${MODEL_SHORT}: rerun DONE"
done

log "Stage-2 rerun complete for both models."
