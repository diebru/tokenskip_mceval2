# tokenskip_mceval

Bridge between [TokenSkip](../TokenSkip/) and [McEval](../McEval/) — runs CoT-eliciting inference on McEval tasks and separates chain-of-thought text (for LLMLingua compression) from code (for McEval's executor).

## Layout

```
tokenskip_mceval/
├── languages.py          # McEval lang → markdown fence tag
├── split_cot_code.py     # split "<cot> ```code``` " → (cot, code)
├── infer_mceval.py       # vLLM inference, writes McEval-compatible jsonl
└── scripts/
    ├── run_baseline_pilot.sh   # 10 Python tasks, sanity check
    ├── run_baseline_full.sh    # all 40 languages
    └── run_eval_docker.sh      # runs McEval eval_all.py in the official Docker image
```

## Server setup

```bash
# Python env
conda create -n tokenskip python=3.12 -y
conda activate tokenskip
pip install vllm transformers

# McEval Docker image (for the eval step only — inference runs on the host)
docker pull multilingualnlp/mceval
```

## Step 1 — pilot (verify CoT is produced)

```bash
cd tokenskip_mceval
bash scripts/run_baseline_pilot.sh
# Check that outputs/.../Python.jsonl has non-empty cot_text fields:
python -c "import json; [print(len(json.loads(l)['cot_text']), '|', json.loads(l)['extracted_code'][:60]) \
  for l in open('../outputs/Qwen2.5-Coder-7B-Instruct/baseline-pilot/Python.jsonl')]"
```

If `cot_text` is mostly empty, the prompt isn't eliciting reasoning — adjust `COT_PROMPT_TEMPLATE` in `infer_mceval.py`.

## Step 2 — full baseline

```bash
bash scripts/run_baseline_full.sh
```

## Step 3 — accuracy via McEval Docker

```bash
RESULTS_DIR=../outputs/Qwen2.5-Coder-7B-Instruct/baseline \
SAVE_DIR=../outputs/Qwen2.5-Coder-7B-Instruct/eval \
bash scripts/run_eval_docker.sh
```

McEval's per-language extractors will pull the code from the `raw_generation` field automatically — they handle markdown code fences, which is the format our prompt elicits.

## Output schema

Each line in `outputs/{model}/{tag}/{Lang}.jsonl` contains all original McEval task fields plus:

- `raw_generation: [string]` — full model output (McEval expects this)
- `cot_text: string` — everything before the final code fence (LLMLingua input)
- `extracted_code: string` — content of the final ```` ``` ```` block

## Next stages (not yet wired)

- LLMLingua compression of `cot_text` at multiple ratios
- LoRA SFT on (instruction, compressed_cot + code) pairs using LLaMA-Factory
- Inference with the LoRA adapter while measuring PDU energy
