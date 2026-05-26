"""CoT-eliciting inference on McEval tasks with vLLM.

Supports all three McEval typologies:
  - generation   : instruction -> code (50/lang, ~2k total)
  - completion   : instruction with [MASK] -> filled-in code (~250/lang, ~10k total)
  - explanation  : code -> docstring (50/lang, ~2k total)

In every case we prompt the model to emit reasoning before the final answer
and then split the two apart, so the CoT text can be compressed by LLMLingua
while the final answer stays untouched for downstream evaluation.

Output layout (per-language jsonl, McEval-compatible):
    {out_dir}/
        Python.jsonl
        Java.jsonl
        ...
The `raw_generation` field is set so that McEval's eval_all.py works
unchanged for generation and completion. For explanation, raw_generation
holds *only* the extracted docstring (since stage-2 plugs it directly into
its code-generation template), and the full model output is preserved in
`full_model_output` for debugging.
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer
import vllm

from languages import LANG_TO_FENCE, jsonl_basename
from split_cot_code import split_cot_code, split_cot_answer


GENERATION_PROMPT = (
    "Solve the following programming task. First, think through your "
    "approach step by step in plain text. Then provide the complete final "
    "solution inside a single ```{fence} ... ``` code block at the end.\n\n"
    "{instruction}"
)

COMPLETION_PROMPT = (
    "Complete the following code. First, think step by step about what "
    "should fill the [MASK] regions and why. Then provide the complete "
    "filled-in function inside a single ```{fence} ... ``` code block at "
    "the end.\n\n"
    "{instruction}"
)

EXPLANATION_PROMPT = (
    "Analyze the following code step by step. First, explain in plain text "
    "what the code does, what algorithm or approach it uses, and what its "
    "inputs and outputs are. Then write the final concise docstring (at most "
    "500 characters) between <answer> and </answer> tags.\n\n"
    "{instruction}"
)


# Per-typology config: data subdir relative to --data-root, prompt template,
# splitter function, and how to populate raw_generation for McEval eval.
TASK_CONFIG = {
    "generation": {
        "data_subdir": "data",
        "prompt_template": GENERATION_PROMPT,
        "splitter": split_cot_code,
        "raw_gen_mode": "full",
    },
    "completion": {
        "data_subdir": "completion/completion_data/merge",
        "prompt_template": COMPLETION_PROMPT,
        "splitter": split_cot_code,
        "raw_gen_mode": "full",
    },
    "explanation": {
        "data_subdir": "explanation/explaination_data",
        "prompt_template": EXPLANATION_PROMPT,
        "splitter": split_cot_answer,
        "raw_gen_mode": "extracted_only",
    },
}


def build_prompts(tokenizer, tasks, fence_tag, template, system_msg):
    prompts = []
    for task in tasks:
        user = template.format(fence=fence_tag, instruction=task["instruction"])
        messages = []
        if system_msg:
            messages.append({"role": "system", "content": system_msg})
        messages.append({"role": "user", "content": user})
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return prompts


def load_tasks(data_dir: Path, lang: str, limit):
    path = data_dir / jsonl_basename(lang)
    with open(path, encoding="utf-8") as f:
        tasks = [json.loads(line) for line in f if line.strip()]
    if limit is not None:
        tasks = tasks[:limit]
    return tasks


def run_language(llm, tokenizer, data_dir, out_dir, lang, sampling_params,
                 system_msg, limit, task_cfg):
    if lang not in LANG_TO_FENCE:
        raise ValueError(f"Unknown McEval language: {lang!r}")
    fence_tag = LANG_TO_FENCE[lang]
    tasks = load_tasks(data_dir, lang, limit)
    prompts = build_prompts(tokenizer, tasks, fence_tag,
                            task_cfg["prompt_template"], system_msg)

    outputs = llm.generate(prompts, sampling_params)
    assert len(outputs) == len(tasks)

    out_path = out_dir / jsonl_basename(lang)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    splitter = task_cfg["splitter"]
    raw_gen_mode = task_cfg["raw_gen_mode"]
    with open(out_path, "w", encoding="utf-8") as f:
        for task, out in zip(tasks, outputs):
            generated = out.outputs[0].text
            cot, answer = splitter(generated)
            record = dict(task)
            if raw_gen_mode == "full":
                # McEval extractor pulls code from the fence in raw_generation
                record["raw_generation"] = [generated]
            elif raw_gen_mode == "extracted_only":
                # Explanation stage-2 plugs raw_generation directly into a
                # code-gen template, so only the docstring text belongs here.
                record["raw_generation"] = [answer]
                record["full_model_output"] = generated
            else:
                raise ValueError(f"Unknown raw_gen_mode {raw_gen_mode!r}")
            record["cot_text"] = cot
            record["extracted_answer"] = answer
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[{lang}] wrote {len(tasks)} -> {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=list(TASK_CONFIG), default="generation",
                   help="McEval task type to run inference on")
    p.add_argument("--model-path", required=True, help="HF id or local path to the model")
    p.add_argument("--data-root", default="../McEval",
                   help="Root of the McEval repo (contains data/, completion/, explanation/)")
    p.add_argument("--out-dir", required=True,
                   help="Where to write per-language jsonl outputs for this task")
    p.add_argument("--languages", nargs="+", default=["Python"],
                   help="McEval language keys, or 'all' for every language")
    p.add_argument("--limit", type=int, default=None,
                   help="Truncate each language to N tasks (for pilots)")
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
    task_cfg = TASK_CONFIG[args.task]
    data_dir = (Path(args.data_root) / task_cfg["data_subdir"]).resolve()
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
        run_language(llm, tokenizer, data_dir, out_dir, lang, sampling_params,
                     args.system_msg, args.limit, task_cfg)


if __name__ == "__main__":
    main()
