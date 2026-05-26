"""CoT-eliciting inference on McEval tasks with vLLM.

For each task in the selected McEval language file(s), we:
  1. Wrap the task's `instruction` with a CoT-eliciting preamble.
  2. Render via the model's chat template.
  3. Generate with vLLM.
  4. Split each output into (cot_text, code) using split_cot_code.
  5. Save per-language jsonl preserving McEval's required schema
     (raw_generation field), plus a `cot_text` field for LLMLingua.

Output layout (McEval-compatible — feed straight to eval_all.py inside Docker):
    {out_dir}/
        Python.jsonl
        Java.jsonl
        ...
"""

import argparse
import json
import os
from pathlib import Path

from transformers import AutoTokenizer
import vllm

from languages import LANG_TO_FENCE, jsonl_basename
from split_cot_code import split_cot_code


COT_PROMPT_TEMPLATE = (
    "Solve the following programming task. First, think through your approach "
    "step by step in plain text. Then provide the complete final solution "
    "inside a single ```{fence} ... ``` code block at the end.\n\n"
    "{instruction}"
)


def build_prompts(tokenizer, tasks, fence_tag, system_msg):
    prompts = []
    for task in tasks:
        user = COT_PROMPT_TEMPLATE.format(fence=fence_tag, instruction=task["instruction"])
        messages = []
        if system_msg:
            messages.append({"role": "system", "content": system_msg})
        messages.append({"role": "user", "content": user})
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return prompts


def load_tasks(data_dir: Path, lang: str, limit: int | None):
    path = data_dir / jsonl_basename(lang)
    with open(path, encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f if line.strip()]
    if limit is not None:
        tasks = tasks[:limit]
    return tasks


def run_language(llm, tokenizer, data_dir, out_dir, lang, sampling_params, system_msg, limit):
    if lang not in LANG_TO_FENCE:
        raise ValueError(f"Unknown McEval language: {lang!r}")
    fence_tag = LANG_TO_FENCE[lang]
    tasks = load_tasks(data_dir, lang, limit)
    prompts = build_prompts(tokenizer, tasks, fence_tag, system_msg)

    outputs = llm.generate(prompts, sampling_params)
    assert len(outputs) == len(tasks)

    out_path = out_dir / jsonl_basename(lang)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for task, out in zip(tasks, outputs):
            generated = out.outputs[0].text
            cot, code = split_cot_code(generated)
            record = dict(task)
            record["raw_generation"] = [generated]
            record["cot_text"] = cot
            record["extracted_code"] = code
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[{lang}] wrote {len(tasks)} → {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True, help="HF id or local path to the model")
    p.add_argument("--data-dir", default="../McEval/data", help="Folder with McEval per-language jsonl")
    p.add_argument("--out-dir", required=True, help="Where to write per-language jsonl outputs")
    p.add_argument("--languages", nargs="+", default=["Python"],
                   help="One or more McEval language keys. Use 'all' for every language.")
    p.add_argument("--limit", type=int, default=None, help="Truncate each language to N tasks (for pilots)")
    p.add_argument("--max-tokens", type=int, default=1536)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--system-msg", default="You are an expert programmer.")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    if args.languages == ["all"]:
        languages = list(LANG_TO_FENCE.keys())
    else:
        languages = args.languages

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm = vllm.LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    sampling_params = vllm.SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    for lang in languages:
        run_language(llm, tokenizer, data_dir, out_dir, lang, sampling_params, args.system_msg, args.limit)


if __name__ == "__main__":
    main()
