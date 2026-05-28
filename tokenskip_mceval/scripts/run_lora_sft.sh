#!/usr/bin/env bash
# Run LoRA SFT for one (model, typology) and merge the adapter at the end.
#
# Usage:
#   bash scripts/run_lora_sft.sh <MODEL_SHORT> <TYPOLOGY> [LLAMAFACTORY_DIR]
# Example:
#   bash scripts/run_lora_sft.sh Qwen2.5-Coder-7B-Instruct generation ~/LLaMA-Factory
#
# Defaults LLAMAFACTORY_DIR to the LlamaFactory clone at ../../LlamaFactory.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_SHORT="${1:?MODEL_SHORT required, e.g. Qwen2.5-Coder-7B-Instruct}"
TYPOLOGY="${2:?TYPOLOGY required: generation | completion | explanation}"
LLAMAFACTORY_DIR="${3:-${LLAMAFACTORY_DIR:-../../LlamaFactory}}"

case "$MODEL_SHORT" in
    *Qwen*)   MODEL_PATH="Qwen/${MODEL_SHORT}"; TEMPLATE="qwen" ;;
    *Llama*)  MODEL_PATH="meta-llama/${MODEL_SHORT}"; TEMPLATE="llama3" ;;
    *) echo "Unknown model: $MODEL_SHORT"; exit 1 ;;
esac

OUT_ROOT="../outputs"
DATASET_DIR="$(realpath ${OUT_ROOT}/${MODEL_SHORT}/train)"
DATASET_NAME="mceval_tokenskip_${TYPOLOGY}"
OUTPUT_DIR="$(realpath -m ${OUT_ROOT}/${MODEL_SHORT}/lora-${TYPOLOGY})"
MERGED_DIR="$(realpath -m ${OUT_ROOT}/${MODEL_SHORT}/merged-${TYPOLOGY})"
CONFIG_TMP="$(mktemp -t lora_sft_XXXXXX.yaml)"

# 1) Register our JSON in a local dataset_info.json that LlamaFactory will read.
cat > "${DATASET_DIR}/dataset_info.json" <<EOF
{
  "mceval_tokenskip_generation": {
    "file_name": "generation.json",
    "formatting": "alpaca",
    "columns": { "prompt": "instruction", "query": "input", "response": "output" }
  },
  "mceval_tokenskip_completion": {
    "file_name": "completion.json",
    "formatting": "alpaca",
    "columns": { "prompt": "instruction", "query": "input", "response": "output" }
  },
  "mceval_tokenskip_explanation": {
    "file_name": "explanation.json",
    "formatting": "alpaca",
    "columns": { "prompt": "instruction", "query": "input", "response": "output" }
  }
}
EOF

if [[ ! -f "${DATASET_DIR}/${TYPOLOGY}.json" ]]; then
    echo "Training file missing: ${DATASET_DIR}/${TYPOLOGY}.json"
    echo "Run scripts/run_build_training.sh first."
    exit 1
fi

# 2) Render the YAML template with this run's values.
sed \
    -e "s|\${MODEL_PATH}|${MODEL_PATH}|g" \
    -e "s|\${DATASET_DIR}|${DATASET_DIR}|g" \
    -e "s|\${DATASET_NAME}|${DATASET_NAME}|g" \
    -e "s|\${OUTPUT_DIR}|${OUTPUT_DIR}|g" \
    -e "s|\${TEMPLATE}|${TEMPLATE}|g" \
    configs/lora_sft.yaml > "${CONFIG_TMP}"

echo "=== rendered config (${CONFIG_TMP}) ==="
cat "${CONFIG_TMP}"
echo "========================================"

# 3) Train.
pushd "${LLAMAFACTORY_DIR}" > /dev/null
llamafactory-cli train "${CONFIG_TMP}"
popd > /dev/null

# 4) Merge LoRA adapter into a standalone model dir for inference.
MERGE_CFG="$(mktemp -t lora_merge_XXXXXX.yaml)"
cat > "${MERGE_CFG}" <<EOF
### model + adapter
model_name_or_path: ${MODEL_PATH}
adapter_name_or_path: ${OUTPUT_DIR}
template: ${TEMPLATE}
finetuning_type: lora
trust_remote_code: true

### export
export_dir: ${MERGED_DIR}
export_size: 5
export_device: cpu
export_legacy_format: false
EOF

pushd "${LLAMAFACTORY_DIR}" > /dev/null
llamafactory-cli export "${MERGE_CFG}"
popd > /dev/null

# Patch tokenizer_config.json that llamafactory-cli export wrote with
# extra_special_tokens=[], which transformers >=4.45 rejects.
python fix_tokenizer_config.py "${MERGED_DIR}"

echo
echo "Done: adapter at ${OUTPUT_DIR}, merged model at ${MERGED_DIR}"
