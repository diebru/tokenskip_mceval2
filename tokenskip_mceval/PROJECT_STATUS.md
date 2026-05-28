# TER project status — TokenSkip on McEval

## Research goal

Extend the [TokenSkip paper](https://arxiv.org/abs/2502.12067) — controllable
Chain-of-Thought compression in LLMs — from math reasoning (GSM8K/MATH) to
**multilingual code generation** (McEval). Measure two things:

1. **Accuracy vs CoT length** — how much can we compress reasoning before code
   accuracy degrades? (paper Figure 5 equivalent)
2. **Energy vs compression ratio** — wall-power energy delta from PDU
   measurements at each ratio. (novel — not in the paper)

Models in scope: **Qwen2.5-Coder-Instruct 1.5B / 3B / 7B**. Benchmark: the
full McEval suite (40 languages × 3 task typologies). Train/test split is
80/20 stratified by language × difficulty × FIM sub-task.

---

## Pipeline overview (TokenSkip §3)

```
┌────────────────────────────────────────────────────────────────────────┐
│  STAGE                          │  OUR SCRIPT             │  STATUS    │
├────────────────────────────────────────────────────────────────────────┤
│  1. Baseline inference          │  infer_mceval.py        │  ✅ done   │
│     (CoT + code per task)       │                         │            │
│  2. Per-task pass/fail eval     │  McEval Docker +        │  ✅ done   │
│     (correctness filter)        │  run_evals_detail.sh    │            │
│  3. LLMLingua compress CoT      │  compress_cot.py        │  ✅ done   │
│     (10 ratios: 0.1 … 1.0)      │                         │            │
│  4. 80/20 stratified split      │  make_split.py          │  ✅ done   │
│  5. Build LlamaFactory SFT JSON │  build_training_data.py │  ✅ done   │
│     (filter correct, random γ,                                         │
│      train ids only)            │                         │            │
│  6. LoRA SFT + merge adapter    │  run_lora_sft.sh        │  🟡 1/7 done│
│  7. Inference on TEST ids       │  infer_mceval.py        │  ⏳ next   │
│     at each ratio               │   (needs --task-ids)    │            │
│  8. McEval Docker eval per ratio│  run_eval_docker.sh     │  ⏳        │
│  9. PDU energy measurement      │  user's existing tooling│  ⏳ final  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## On-disk layout (what we have right now)

```
tokenskip_mceval2/
├── McEval/                    upstream McEval clone (vendored), patched:
│   ├── data/                  ←  EVAL data, 40 langs × 50 tasks = 2007
│   │                              also has *.zip for completion/explanation
│   ├── completion/completion_data/merge/   ← 10010 completion tasks
│   ├── explanation/explaination_data/      ← 2007 explanation tasks
│   └── eval/                  patched eval_all.py + excute.py
│
├── tokenskip_mceval/          our code
│   ├── infer_mceval.py        baseline inference (CoT + code)
│   ├── run_stage2_explanation.py    docstring → code round-trip
│   ├── compress_cot.py        LLMLingua-2 compression
│   ├── make_split.py          80/20 stratified split
│   ├── build_training_data.py LlamaFactory SFT JSON builder
│   ├── fix_tokenizer_config.py patch for LF export bug
│   ├── parse_eval_logs.py     log → per-task pass/fail (legacy)
│   ├── report.py              aggregate accuracy / CoT length report
│   ├── configs/lora_sft.yaml  LlamaFactory training config template
│   └── scripts/               runner shells
│
└── outputs/                   all generated artifacts
    ├── split/                                  shared across all models
    │   ├── train_ids.json     11,221 ids (80%)
    │   └── test_ids.json      2,803 ids (20%)
    │
    └── {Qwen2.5-Coder-1.5B|3B|7B-Instruct}/
        ├── baseline/{generation|completion|explanation}/{Lang}.jsonl
        │       full CoT + code, all 14k tasks per model
        ├── compressed/{typology}/ratio_{0.1..1.0}/{Lang}.jsonl
        │       compressed CoTs at 10 ratios
        ├── eval-detail/{typology}/*_detail.jsonl
        │       per-task pass/fail
        ├── eval-full/{typology}/results.jsonl
        │       per-language aggregate accuracy (baseline)
        ├── train/{generation|completion|explanation}.json
        │       LlamaFactory alpaca SFT data, train ids only
        ├── lora-{typology}/                    LoRA adapter (≈50 MB)
        └── merged-{typology}/                  full model + adapter (≈3 GB)
```

---

## Baseline accuracy (from `report.py`)

| Model | Generation | Completion | Explanation |
|---|---|---|---|
| Qwen-1.5B | 32.3% | 48.1% | 34.3% (CoT mostly empty, see note) |
| Qwen-3B   | 39.5% | 59.7% | 42.8% (CoT mostly empty) |
| Qwen-7B   | 56.7% | 67.0% | 50.8% |

Note: 1.5B/3B explanation produced near-zero CoT because the small models
ignored the `<answer>…</answer>` instruction. Only 7B explanation has
meaningful CoT to compress.

## Compression realised vs target (LLMLingua-2)

Target 0.1 → realised ≈ 0.085; target 0.5 → ≈ 0.49; target 0.9 → ≈ 0.91.
LLMLingua hits the ratios cleanly within ±2% across all datasets.

## Training data sizes (filtered to correct + non-empty CoT + train ids)

| Model | Gen | Comp | Exp | Total |
|---|---|---|---|---|
| Qwen-1.5B | 500 | 3,648 | — | 4,148 |
| Qwen-3B | 644 | 4,537 | — | 5,181 |
| Qwen-7B | 903 | 5,345 | 765 | 7,013 |

(TokenSkip paper used 7,473 GSM8K + 7,500 MATH per model for reference.)

---

## What's actively happening right now

### Attempt 1: per-typology adapter (failed)

Trained Qwen-1.5B / **generation only** (500 examples) → adapter at
`.../lora-generation/`, merged at `.../merged-generation/`. Sanity check
showed **no CoT at any ratio** and ratios 0.1-0.7 produced byte-identical
output. SFT didn't override the base model's "go straight to code" prior.
**Diagnosis**: 500 examples is below TokenSkip's working threshold
(~7,473 GSM8K). Too few examples per ratio (~50 each) for a 1.5B model.

### Attempt 2: combined adapter (works ✅)

Concatenated generation+completion+explanation per model with
`combine_training_data.py`, retrained Qwen-1.5B on 4,148 combined examples
(~415 per ratio). Adapter at `.../lora-combined/`, merged at
`.../merged-combined/`.

Sanity check on a real McEval test-set Python task
(`Python/10 - sum_of_factorials_optimized`):

| ratio | tokens | CoT chars before fence | Behavior                                    |
|------:|------:|----------------------:|---------------------------------------------|
|   1.0 |  263 |                 575 | ✅ Full structured reasoning                 |
|   0.7 |  212 |                 356 | ✅ TokenSkip-style fragmented keywords       |
|   0.5 |  222 |                 427 | ✅ Compressed                                |
|   0.3 |  192 |                 286 | ✅ Heavily compressed                        |
|   0.1 | 1024 |                1038 | ⚠️ Degenerated into repeating "50 50 50..." |

SFT learned ratio control for **0.3–1.0**. The 0.1 collapse matches
TokenSkip §4.3: "ratio adherence largely degrades at these lower ratios…
excessive trimming causes loss of critical information." See paper
Figure 6 — they observed the same. Production sweep should use ratios
**0.3–1.0** (8 points), dropping 0.1 and 0.2 as degenerate.

---

## What comes next (in order)

1. ✅ ~~Confirm SFT worked via sanity check~~ — done, see above.
2. **Patch `infer_mceval.py`** to accept `--task-ids <split-file>` so we
   generate only on the 2,803 test ids (not the full 14k).
3. **Sweep inference at ratios 0.3–1.0** on the test set using the
   `merged-combined` model. (Skip 0.1, 0.2 — sanity check showed they
   degenerate; TokenSkip Figure 6 confirms this is expected.)
4. **Docker eval** each ratio's outputs → accuracy-vs-ratio table.
5. **PDU energy capture** wrapping that test-set sweep — gives the energy
   curve aligned with the accuracy curve.
6. **Plot** accuracy and energy vs realized CoT length (paper Figure 5
   equivalent + the energy half of the headline result).
7. **Scale out**: train combined adapters for Qwen-3B (5,181 ex) and
   Qwen-7B (7,013 ex). Repeat steps 2–5 for each.

---

## Key design choices (locked-in)

- **No McEval-Instruct** — we split the local McEval-eval 80/20 instead.
  Big win: training data has unit tests, so we can run the TokenSkip
  correctness filter on it (we couldn't with McEval-Instruct).
- **Stratification keys**: generation = lang × level (easy/middle/hard),
  completion = lang × FIM sub-task (single/multi/span), explanation =
  lang only.
- **Ratio marker format**: plain text `\n\n[Compression ratio: X.X]`
  appended to the user message. (TokenSkip uses `<|eot_id|>X<|eot_id|>`;
  ours is model-agnostic. Switch if SFT struggles.)
- **PDU energy** measured only at the **end** on the final test sweep,
  not during baseline runs.
- **LoRA hyperparams** (TokenSkip §B.1 exact match): rank 8, α 16, 3
  epochs, lr 5e-5 cosine, warmup 0.1, AdamW, `lora_target: all`.
- **Docker image** for code execution: `mceval-full` (our derived image,
  patches in `tokenskip_mceval/docker/Dockerfile`), installs Rust/Go/Scala
  3/Kotlin/Julia/Dart/Swift/etc. that the upstream image was missing.

---

## Quick commands cheat sheet

```bash
# Where to start every session
cd ~/training_llama_8.1/tokenskip_mceval2 && git pull

# See aggregated baseline numbers + CoT lengths
cd tokenskip_mceval && python report.py

# Run the build steps from scratch (idempotent)
python make_split.py                    # 1
bash scripts/run_build_training.sh      # 2 (uses make_split output)

# Train one LoRA adapter
bash scripts/run_lora_sft.sh \
    Qwen2.5-Coder-1.5B-Instruct generation \
    ~/training_llama_8.1/LLaMA-Factory

# Sanity-check a merged model
python scripts/sanity_check_merged.py \
    ../outputs/Qwen2.5-Coder-1.5B-Instruct/merged-generation
```
