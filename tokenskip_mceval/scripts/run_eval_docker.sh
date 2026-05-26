#!/usr/bin/env bash
# Run McEval's eval_all.py inside the multilingualnlp/mceval Docker image
# against a folder of per-language jsonl outputs produced by infer_mceval.py.
#
# Usage:
#   RESULTS_DIR=../outputs/Qwen2.5-Coder-7B-Instruct/baseline-pilot \
#   SAVE_DIR=../outputs/Qwen2.5-Coder-7B-Instruct/eval-pilot \
#   bash run_eval_docker.sh
set -euo pipefail

RESULTS_DIR="${RESULTS_DIR:?set RESULTS_DIR to a folder of per-language jsonl}"
SAVE_DIR="${SAVE_DIR:?set SAVE_DIR to where eval results should be written}"
MCEVAL_REPO="${MCEVAL_REPO:-$(cd ../McEval && pwd)}"
IMAGE="${IMAGE:-multilingualnlp/mceval}"

RESULTS_DIR="$(cd "$RESULTS_DIR" && pwd)"
mkdir -p "$SAVE_DIR"
SAVE_DIR="$(cd "$SAVE_DIR" && pwd)"

docker run --rm \
    -v "$MCEVAL_REPO":/workspace/McEval \
    -v "$RESULTS_DIR":/workspace/results \
    -v "$SAVE_DIR":/workspace/eval_out \
    -w /workspace/McEval/eval \
    "$IMAGE" \
    python -u eval_all.py \
        --result_path /workspace/results \
        --save_path /workspace/eval_out
