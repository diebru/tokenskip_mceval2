"""Post-SFT inference: run the merged TokenSkip model on the held-out
TEST split at a single compression ratio.

Different from infer_mceval.py:
  - Uses the RAW McEval instruction (no CoT-eliciting preamble) and
    appends "[Compression ratio: X.X]" - matches the SFT training format
    produced by build_training_data.py.
  - Filters tasks to those in --task-ids-file (typically test_ids.json).
  - Outputs in McEval's expected schema so the existing Docker eval
    (run_eval_docker.sh) scores it unchanged.

Usage:
    python infer_post_sft.py \\
        --model-path ../outputs/Qwen2.5-Coder-1.5B-Instruct/merged-combined \\
        --task generation \\
        --task-ids-file ../outputs/split/test_ids.json \\
        --ratio 0.5 \\
        --out-dir ../outputs/Qwen2.5-Coder-1.5B-Instruct/test-sweep/ratio_0.5/generation \\
        --languages all
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer
import vllm

from languages import LANG_TO_FENCE, jsonl_basename
from split_cot_code import split_cot_code, split_cot_answer

# Per-typology data dir under --data-root (matches infer_mceval.py)
DATA_SUBDIR = {
    "generation": "data",
    "completion": "completion/completion_data/merge",
    "explanation": "explanation/explaination_data",
}

RATIO_MARKER = "\n\n[Compression ratio: {ratio:.1f}]"


def load_task_ids(path: Path, typology: str) -> set:
    with open(path) as f:
        data = json.load(f)
    ids = set(data.get(typology, []))
    if not ids:
        raise SystemExit(f"No ids for {typology!r} in {path}")
    return ids


def load_tasks(data_dir: Path, lang: str, keep_ids: set):
    path = data_dir / jsonl_basename(lang)
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["task_id"] in keep_ids:
                out.append(r)
    return out


def build_prompt(tokenizer, instruction: str, ratio: float, system_msg: str) -> str:
    user = instruction + RATIO_MARKER.format(ratio=ratio)
    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": user})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--task", required=True, choices=list(DATA_SUBDIR))
    ap.add_argument("--task-ids-file", required=True,
                    help="JSON {typology: [ids]} (e.g. test_ids.json)")
    ap.add_argument("--ratio", type=float, required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--data-root", default="../McEval")
    ap.add_argument("--languages", nargs="+", default=["all"])
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--system-msg", default="You are an expert programmer.")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = ap.parse_args()

    keep_ids = load_task_ids(Path(args.task_ids_file), args.task)
    data_dir = (Path(args.data_root) / DATA_SUBDIR[args.task]).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    languages = list(LANG_TO_FENCE) if args.languages == ["all"] else args.languages
    splitter = split_cot_answer if args.task == "explanation" else split_cot_code
    raw_gen_mode = "extracted_only" if args.task == "explanation" else "full"

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

    grand_total = 0
    for lang in languages:
        tasks = load_tasks(data_dir, lang, keep_ids)
        if not tasks:
            continue
        prompts = [build_prompt(tokenizer, t["instruction"], args.ratio,
                                args.system_msg) for t in tasks]
        outputs = llm.generate(prompts, sampling_params)
        out_path = out_dir / jsonl_basename(lang)
        with open(out_path, "w", encoding="utf-8") as f:
            for task, out in zip(tasks, outputs):
                generated = out.outputs[0].text
                cot, answer = splitter(generated)
                rec = dict(task)
                if raw_gen_mode == "full":
                    rec["raw_generation"] = [generated]
                else:
                    rec["raw_generation"] = [answer]
                    rec["full_model_output"] = generated
                rec["cot_text"] = cot
                rec["extracted_answer"] = answer
                rec["compression_ratio_requested"] = args.ratio
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{lang}] {len(tasks):>4} tasks at ratio {args.ratio} -> {out_path}")
        grand_total += len(tasks)
    print(f"Total: {grand_total} tasks written under {out_dir}")


if __name__ == "__main__":
    main()
