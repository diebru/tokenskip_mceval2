#!/usr/bin/env bash
# Re-run all evals with per-task detail capture (eval_all.py now writes
# *_detail.jsonl). Runs in Docker on CPU only — no GPU — so it can run in
# the background alongside GPU work (e.g. LLMLingua compression).
#
# Writes to a fresh eval-detail/ dir per (model, typology) so every language
# runs (the skip-already-done logic keys off results.jsonl in the save dir).
#
# Launch:  nohup bash scripts/run_evals_detail.sh > evals_detail.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

OUT_ROOT="../outputs"
IMAGE="${IMAGE:-mceval-full}"

# (model_short, baseline_subdir) pairs. explanation uses the stage-2 outputs
# (round-trip code) since that's what gets executed.
JOBS=(
    "Qwen2.5-Coder-7B-Instruct|generation|generation"
    "Qwen2.5-Coder-7B-Instruct|completion|completion"
    "Qwen2.5-Coder-7B-Instruct|explanation|explanation_stage2"
    "Qwen2.5-Coder-3B-Instruct|generation|generation"
    "Qwen2.5-Coder-3B-Instruct|completion|completion"
    "Qwen2.5-Coder-3B-Instruct|explanation|explanation_stage2"
    "Qwen2.5-Coder-1.5B-Instruct|generation|generation"
    "Qwen2.5-Coder-1.5B-Instruct|completion|completion"
    "Qwen2.5-Coder-1.5B-Instruct|explanation|explanation_stage2"
)

log() { echo -e "\n=== [$(date '+%H:%M:%S')] $* ==="; }

for entry in "${JOBS[@]}"; do
    IFS="|" read -r MODEL TYP SUBDIR <<< "$entry"
    RESULTS="${OUT_ROOT}/${MODEL}/baseline/${SUBDIR}"
    SAVE="${OUT_ROOT}/${MODEL}/eval-detail/${TYP}"

    if [[ ! -d "$RESULTS" ]] || [[ -z "$(ls -A "$RESULTS"/*.jsonl 2>/dev/null)" ]]; then
        log "${MODEL}/${TYP}: no baseline outputs at ${RESULTS}, skipping"
        continue
    fi

    log "${MODEL}/${TYP}: detail eval -> ${SAVE}"
    IMAGE="$IMAGE" RESULTS_DIR="$RESULTS" SAVE_DIR="$SAVE" \
        bash scripts/run_eval_docker.sh
done

log "All detail evals complete."
