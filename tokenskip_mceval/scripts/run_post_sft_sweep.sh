#!/usr/bin/env bash
# Sweep ratios 0.3 - 1.0 on the test split with a TokenSkip-trained model,
# then McEval Docker eval each ratio.
#
# Usage:
#   bash scripts/run_post_sft_sweep.sh <MODEL_SHORT> [TYPOLOGY]
#
# Examples:
#   bash scripts/run_post_sft_sweep.sh Qwen2.5-Coder-1.5B-Instruct generation
#   bash scripts/run_post_sft_sweep.sh Qwen2.5-Coder-7B-Instruct completion
#
# Defaults to typology=generation (fastest: ~403 tasks * 8 ratios).
# Ratios 0.1, 0.2 are dropped per TokenSkip §4.3 (degenerate).
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL_SHORT="${1:?MODEL_SHORT required}"
TYPOLOGY="${2:-generation}"
SUFFIX="${3:-combined}"   # which merged-{SUFFIX} dir to use; matches LoRA name
RATIOS="${RATIOS:-0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0}"
TP="${TP:-1}"

OUT_ROOT="../outputs"
MERGED_DIR="${OUT_ROOT}/${MODEL_SHORT}/merged-${SUFFIX}"
SPLIT_FILE="${OUT_ROOT}/split/test_ids.json"
SWEEP_ROOT="${OUT_ROOT}/${MODEL_SHORT}/test-sweep-${SUFFIX}"
EVAL_ROOT="${OUT_ROOT}/${MODEL_SHORT}/eval-test-sweep-${SUFFIX}"

if [[ ! -d "$MERGED_DIR" ]]; then
    echo "Merged model not found at $MERGED_DIR. Run run_lora_sft.sh ... combined first."
    exit 1
fi
if [[ ! -f "$SPLIT_FILE" ]]; then
    echo "Test ids file not found at $SPLIT_FILE. Run make_split.py first."
    exit 1
fi

log() { echo -e "\n=== [$(date '+%H:%M:%S')] $* ==="; }

# 1) Inference at each ratio (GPU)
for r in $RATIOS; do
    OUT="${SWEEP_ROOT}/ratio_${r}/${TYPOLOGY}"
    if compgen -G "${OUT}/*.jsonl" > /dev/null; then
        log "${MODEL_SHORT}/${TYPOLOGY} ratio=${r}: outputs exist, skipping"
        continue
    fi
    log "${MODEL_SHORT}/${TYPOLOGY} ratio=${r}: inference"
    python infer_post_sft.py \
        --model-path "$MERGED_DIR" \
        --task "$TYPOLOGY" \
        --task-ids-file "$SPLIT_FILE" \
        --ratio "$r" \
        --out-dir "$OUT" \
        --languages all \
        --tensor-parallel-size "$TP"
done

# 2) Docker eval each ratio (CPU)
for r in $RATIOS; do
    OUT="${SWEEP_ROOT}/ratio_${r}/${TYPOLOGY}"
    SAVE="${EVAL_ROOT}/${TYPOLOGY}/ratio_${r}"
    if [[ -f "${SAVE}/$(basename ${TYPOLOGY})_detail.jsonl" ]]; then
        log "ratio=${r}: eval already present, skipping"
        continue
    fi
    log "ratio=${r}: Docker eval"
    IMAGE=mceval-full RESULTS_DIR="$OUT" SAVE_DIR="$SAVE" \
        bash scripts/run_eval_docker.sh
done

log "Post-SFT sweep done for ${MODEL_SHORT}/${TYPOLOGY}"
echo "Outputs:  ${SWEEP_ROOT}/ratio_*/${TYPOLOGY}/"
echo "Eval:     ${EVAL_ROOT}/${TYPOLOGY}/ratio_*/results.jsonl"
echo "Use plot_accuracy_curve.py to render the curve."
