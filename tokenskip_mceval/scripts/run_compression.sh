#!/usr/bin/env bash
# Compress CoTs for all viable (model, typology) datasets with LLMLingua-2.
# Runs on GPU but the model is a small RoBERTa, so it coexists fine with the
# CPU-only detail eval running in the background.
#
# Skips explanation for 1.5B/3B (CoT was ~empty there — nothing to compress).
#
# Launch:  nohup bash scripts/run_compression.sh > compression.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

LLMLINGUA_PATH="${LLMLINGUA_PATH:-microsoft/llmlingua-2-xlm-roberta-large-meetingbank}"
RATIOS="${RATIOS:-0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0}"

log() { echo -e "\n=== [$(date '+%H:%M:%S')] $* ==="; }

log "7B: generation, completion, explanation"
python compress_cot.py --model Qwen2.5-Coder-7B-Instruct \
    --typologies generation completion explanation \
    --ratios $RATIOS --llmlingua-path "$LLMLINGUA_PATH"

log "3B: generation, completion (explanation CoT too sparse)"
python compress_cot.py --model Qwen2.5-Coder-3B-Instruct \
    --typologies generation completion \
    --ratios $RATIOS --llmlingua-path "$LLMLINGUA_PATH"

log "1.5B: generation, completion (explanation CoT too sparse)"
python compress_cot.py --model Qwen2.5-Coder-1.5B-Instruct \
    --typologies generation completion \
    --ratios $RATIOS --llmlingua-path "$LLMLINGUA_PATH"

log "Compression complete."
