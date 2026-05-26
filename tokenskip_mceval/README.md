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

## Three task typologies

McEval has three task types. The inference script supports all of them via
`--task generation|completion|explanation`:

| Task | Input | Final-answer delimiter | Total tasks |
|------|-------|------------------------|-------------|
| `generation` | instruction | ```` ```<lang> ... ``` ```` | 2,007 |
| `completion` | instruction with `[MASK]` regions | ```` ```<lang> ... ``` ```` | 10,128 |
| `explanation` | code | `<answer> ... </answer>` | 2,066 |

The Completion and Explanation data live in zips under `McEval/`. Unzip
once on the server:

```bash
cd McEval/completion && unzip -q completion_data.zip
cd ../explanation && unzip -q explaination_data.zip
cd ../..
```

## Step 1 — pilots (verify CoT for each typology)

Each pilot runs 10 Python tasks (~1 min) to confirm the prompt elicits
reasoning and the splitter extracts the final answer correctly.

```bash
cd tokenskip_mceval
bash scripts/run_baseline_pilot.sh         # generation
bash scripts/run_pilot_completion.sh       # completion
bash scripts/run_pilot_explanation.sh      # explanation

# Inspect cot_text and extracted_answer:
python -c "import json; \
  [print(len(json.loads(l)['cot_text']), '|', json.loads(l)['extracted_answer'][:80]) \
   for l in open('../outputs/Qwen2.5-Coder-7B-Instruct/baseline-pilot/explanation/Python.jsonl')]"
```

## Step 2 — full baselines per typology

```bash
bash scripts/run_baseline_full.sh           # generation, ~30 min
bash scripts/run_baseline_completion.sh     # completion, ~2-3 h
bash scripts/run_baseline_explanation.sh    # explanation, ~30 min (stage 1 only)
```

## Step 3 — accuracy via McEval Docker

```bash
# Generation
IMAGE=mceval-full \
RESULTS_DIR=../outputs/Qwen2.5-Coder-7B-Instruct/baseline/generation \
SAVE_DIR=../outputs/Qwen2.5-Coder-7B-Instruct/eval/generation \
bash scripts/run_eval_docker.sh

# Completion (same eval pipeline, code execution)
IMAGE=mceval-full \
RESULTS_DIR=../outputs/Qwen2.5-Coder-7B-Instruct/baseline/completion \
SAVE_DIR=../outputs/Qwen2.5-Coder-7B-Instruct/eval/completion \
bash scripts/run_eval_docker.sh
```

McEval's per-language extractors pull code from the `raw_generation` field
automatically. Explanation eval is a **two-stage** process — see below.

## Explanation stage 2 (round-trip)

McEval evaluates explanation quality by *round-trip*: take the generated
docstring, feed it to a code-generation prompt, execute the result against
the same tests. If the docstring conveyed enough information, the
regenerated code passes. The stage-2 orchestrator is `run_stage2_explanation.py`
(TODO: not yet written).

## Output schema

Each line in `outputs/{model}/{tag}/{Lang}.jsonl` contains all original McEval task fields plus:

- `raw_generation: [string]` — full model output (McEval expects this)
- `cot_text: string` — everything before the final code fence (LLMLingua input)
- `extracted_code: string` — content of the final ```` ``` ```` block

## Next stages (not yet wired)

- LLMLingua compression of `cot_text` at multiple ratios
- LoRA SFT on (instruction, compressed_cot + code) pairs using LLaMA-Factory
- Inference with the LoRA adapter while measuring PDU energy
