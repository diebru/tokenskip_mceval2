#!/usr/bin/env bash
# Diagnostic: run the BASE model (no SFT) at ratio=1.0 on the test split
# using the same prompt format the SFT'd model gets. If accuracy here is
# ~32% then SFT is the regression cause. If it's ~17% then the prompt
# format change accounts for the gap and SFT preserved accuracy.
#
# Usage:
#   bash scripts/diagnose_prompt_vs_sft.sh <MODEL_SHORT>
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL_SHORT="${1:?MODEL_SHORT required}"
TYPOLOGY="${2:-generation}"
BASE_MODEL="Qwen/${MODEL_SHORT}"
OUT_ROOT="../outputs"
SPLIT_FILE="${OUT_ROOT}/split/test_ids.json"
DIAG_OUT="${OUT_ROOT}/${MODEL_SHORT}/diagnostic-base-newprompt/${TYPOLOGY}"
DIAG_EVAL="${OUT_ROOT}/${MODEL_SHORT}/diagnostic-base-newprompt-eval/${TYPOLOGY}"

log() { echo -e "\n=== [$(date '+%H:%M:%S')] $* ==="; }

if [[ ! -d "$DIAG_OUT" ]] || ! compgen -G "${DIAG_OUT}/*.jsonl" > /dev/null; then
    log "Base ${MODEL_SHORT} @ ratio=1.0 with SFT prompt format"
    python infer_post_sft.py \
        --model-path "$BASE_MODEL" \
        --task "$TYPOLOGY" \
        --task-ids-file "$SPLIT_FILE" \
        --ratio 1.0 \
        --out-dir "$DIAG_OUT" \
        --languages all \
        --tensor-parallel-size 1
fi

log "Docker eval on diagnostic outputs"
IMAGE=mceval-full RESULTS_DIR="$DIAG_OUT" SAVE_DIR="$DIAG_EVAL" \
    bash scripts/run_eval_docker.sh

log "Compute accuracy"
python -c "
import json, glob
total=correct=0
for f in glob.glob('${DIAG_EVAL}/*results.jsonl'):
    for line in open(f):
        if '\t' not in line: continue
        d = json.loads(line.split('\t', 1)[1])
        correct += d.get('correct', 0)
        total += d.get('total_count', 0)
print(f'BASE model + new prompt: {correct}/{total} = {100*correct/total:.2f}%' if total else 'no eval')
"
