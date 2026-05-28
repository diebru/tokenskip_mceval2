"""Sanity check the SFT'd merged model with an ACTUAL McEval task,
matching the prompt format the model saw during training.

The previous version used a hand-written one-liner, which is a different
distribution from the McEval instructions — the model just defaulted to
its base behavior.

Usage:
    python scripts/sanity_check_merged.py \\
        ../outputs/Qwen2.5-Coder-1.5B-Instruct/merged-generation \\
        [task_id]    # e.g. "Python/3"; defaults to the first test-set
                     # Python generation task
"""

import json
import sys
from pathlib import Path

import vllm
from transformers import AutoTokenizer


def find_task(task_id: str | None):
    """Return (record, language) for a task we can re-prompt cleanly."""
    split_path = Path("../outputs/split/test_ids.json")
    if not split_path.exists():
        raise SystemExit(f"missing {split_path}; run make_split.py first")
    test_ids = json.load(open(split_path))
    if task_id is None:
        # Pick the first Python test-set generation task we have a record for.
        py_ids = [t for t in test_ids["generation"] if t.startswith("Python/")]
        if not py_ids:
            raise SystemExit("no Python generation task in test split")
        task_id = py_ids[0]
        print(f"(using default test task: {task_id})")

    lang = task_id.split("/", 1)[0]
    base = Path("../McEval/data") / f"{lang}.jsonl"
    if not base.exists():
        raise SystemExit(f"missing {base}")
    for line in open(base, encoding="utf-8"):
        r = json.loads(line)
        if r["task_id"] == task_id:
            return r, lang
    raise SystemExit(f"task_id {task_id} not found in {base}")


def main():
    merged = sys.argv[1] if len(sys.argv) > 1 else \
        "../outputs/Qwen2.5-Coder-1.5B-Instruct/merged-generation"
    explicit_id = sys.argv[2] if len(sys.argv) > 2 else None

    task, lang = find_task(explicit_id)
    print(f"\n=== Sanity check on {merged} ===")
    print(f"=== Task: {task['task_id']} ({lang}) ===\n")

    # Match the build_training_data.py prompt: the model's user-message in
    # training was instruction + "\n\n[Compression ratio: X.X]"
    base_instruction = task["instruction"]
    print(f"--- instruction (first 200 chars) ---\n{base_instruction[:200]}\n...")

    tok = AutoTokenizer.from_pretrained(merged, trust_remote_code=True)
    llm = vllm.LLM(model=merged, tensor_parallel_size=1, dtype="auto",
                   gpu_memory_utilization=0.7, trust_remote_code=True)

    for ratio in (1.0, 0.7, 0.5, 0.3, 0.1):
        user = f"{base_instruction}\n\n[Compression ratio: {ratio}]"
        messages = [
            {"role": "system", "content": "You are an expert programmer."},
            {"role": "user", "content": user},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        out = llm.generate([prompt], vllm.SamplingParams(temperature=0.0, max_tokens=1024))[0]
        text = out.outputs[0].text
        n_tok = len(out.outputs[0].token_ids)
        # Crude CoT-vs-code split: everything before the first fenced block.
        fence_at = text.find("```")
        cot_chars = fence_at if fence_at >= 0 else len(text)
        print(f"========= ratio={ratio:<4}  {n_tok:>4} tok  "
              f"({cot_chars} chars before code fence) =========")
        print(text[:600])
        print()


if __name__ == "__main__":
    main()
