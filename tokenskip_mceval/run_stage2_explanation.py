"""Stage-2 round-trip for Explanation eval.

For each Explanation task, Stage 1 produced a docstring describing what the
canonical solution does. McEval's official methodology evaluates explanation
quality by feeding that docstring into a *code generation* prompt and
running the resulting code against the original tests. If the generated
code passes, the docstring conveyed enough information.

This script does that stage-2 generation step. Input is the stage-1
outputs (per-language jsonl); output is in McEval's standard generation
schema so the existing Docker eval pipeline scores it unchanged.

Usage:
    python run_stage2_explanation.py \
        --model-path Qwen/Qwen2.5-Coder-7B-Instruct \
        --stage1-dir ../outputs/Qwen2.5-Coder-7B-Instruct/baseline/explanation \
        --out-dir ../outputs/Qwen2.5-Coder-7B-Instruct/baseline/explanation_stage2
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer
import vllm

from languages import LANG_TO_FENCE, jsonl_basename
from split_cot_code import split_cot_code


# Templates lifted from McEval/inference/gen_stage2_instruction.py
TEMPLATE_DEFAULT = (
    "Write a {lang} function `{signature}` to solve the following problem:\n"
    "{docstring}"
)
TEMPLATE_AWK = "Using the awk command in Linux, complete the following task:\n{docstring}"
TEMPLATE_HTML = "Generate HTML code according to the following requirements:\n{docstring}"
TEMPLATE_JSON = "Create a JSON object according to the following requirements:\n{docstring}"
TEMPLATE_MD = "Generate Markdown code according to the following requirements:\n{docstring}"

# Wrap the stage-2 base instruction with the same CoT-eliciting preamble we
# use for ordinary generation, so the model produces reasoning + code fence.
COT_WRAP = (
    "Solve the following programming task. First, think through your "
    "approach step by step in plain text. Then provide the complete final "
    "solution inside a single ```{fence} ... ``` code block at the end.\n\n"
    "{instruction}"
)


def base_instruction(lang: str, signature: str, docstring: str) -> str:
    key = lang.lower()
    if key == "awk":
        return TEMPLATE_AWK.format(docstring=docstring)
    if key == "html":
        return TEMPLATE_HTML.format(docstring=docstring)
    if key == "json":
        return TEMPLATE_JSON.format(docstring=docstring)
    if key == "markdown":
        return TEMPLATE_MD.format(docstring=docstring)
    return TEMPLATE_DEFAULT.format(lang=lang, signature=signature, docstring=docstring)


def build_prompts(tokenizer, stage1_records, lang, system_msg):
    fence = LANG_TO_FENCE[lang]
    prompts = []
    for r in stage1_records:
        # stage 1 set raw_generation = [docstring]
        docstring = r["raw_generation"][0]
        sig = r.get("signature", "")
        base = base_instruction(lang, sig, docstring)
        user = COT_WRAP.format(fence=fence, instruction=base)
        messages = []
        if system_msg:
            messages.append({"role": "system", "content": system_msg})
        messages.append({"role": "user", "content": user})
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return prompts


def run_language(llm, tokenizer, stage1_path: Path, out_path: Path, lang: str,
                 sampling_params, system_msg):
    with open(stage1_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    prompts = build_prompts(tokenizer, records, lang, system_msg)
    outputs = llm.generate(prompts, sampling_params)
    assert len(outputs) == len(records)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r, out in zip(records, outputs):
            generated = out.outputs[0].text
            cot, code = split_cot_code(generated)
            # Preserve all stage-1 fields and overlay the stage-2 ones.
            # Hand-listing fields broke because some McEval languages
            # (AWK, HTML, JSON, Markdown — the file-compare ones) omit
            # 'test' and other "code-task" fields entirely.
            stage1_docstring = r["raw_generation"][0]
            stage1_cot = r.get("cot_text", "")
            record = dict(r)
            record["raw_generation"] = [generated]
            record["cot_text"] = cot
            record["extracted_answer"] = code
            record["stage1_docstring"] = stage1_docstring
            record["stage1_cot_text"] = stage1_cot
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[{lang}] wrote {len(records)} -> {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--stage1-dir", required=True,
                   help="Directory of stage-1 explanation outputs (per-lang jsonl)")
    p.add_argument("--out-dir", required=True,
                   help="Where to write stage-2 outputs (McEval-compatible)")
    p.add_argument("--languages", nargs="+", default=["all"])
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
    stage1_dir = Path(args.stage1_dir).resolve()
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
        stage1_path = stage1_dir / jsonl_basename(lang)
        if not stage1_path.exists():
            print(f"[{lang}] no stage-1 file at {stage1_path}, skipping")
            continue
        out_path = out_dir / jsonl_basename(lang)
        run_language(llm, tokenizer, stage1_path, out_path, lang,
                     sampling_params, args.system_msg)


if __name__ == "__main__":
    main()
