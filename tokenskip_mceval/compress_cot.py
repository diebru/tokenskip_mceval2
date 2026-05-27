"""Compress the cot_text field of baseline outputs with LLMLingua-2.

Mirrors the TokenSkip pipeline (TokenSkip/LLMLingua.py) but for our McEval
per-language outputs. We compress every task's CoT at several ratios; the
correctness filter is applied later when building the SFT training set, so
this step is independent of the eval-detail re-run.

Input:   {out_root}/{model}/baseline/{typology}/{Lang}.jsonl   (has cot_text)
Output:  {out_root}/{model}/compressed/{typology}/ratio_{r}/{Lang}.jsonl

Each output record keeps task_id + instruction + extracted_answer and adds:
    compressed_cot, original_cot_tokens, compressed_cot_tokens, compression_rate

Usage:
    python compress_cot.py \
        --model Qwen2.5-Coder-7B-Instruct \
        --typologies generation completion explanation \
        --ratios 0.5 0.6 0.7 0.8 \
        --llmlingua-path microsoft/llmlingua-2-xlm-roberta-large-meetingbank
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm
from llmlingua import PromptCompressor

OUT_ROOT = Path("../outputs")
# Min CoT length (chars) worth compressing. Below this LLMLingua adds noise
# and there's nothing to gain; we pass the original through unchanged.
MIN_COT_CHARS = 40


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compress_language(llm_lingua, tasks, ratio):
    out = []
    for t in tasks:
        cot = t.get("cot_text", "") or ""
        if len(cot) < MIN_COT_CHARS:
            rec = {
                "task_id": t["task_id"],
                "instruction": t.get("instruction", ""),
                "extracted_answer": t.get("extracted_answer", ""),
                "cot_text": cot,
                "compressed_cot": cot,
                "original_cot_tokens": None,
                "compressed_cot_tokens": None,
                "compression_rate": 1.0,
                "skipped": True,
            }
            out.append(rec)
            continue
        c = llm_lingua.compress_prompt(cot, rate=ratio)
        out.append({
            "task_id": t["task_id"],
            "instruction": t.get("instruction", ""),
            "extracted_answer": t.get("extracted_answer", ""),
            "cot_text": cot,
            "compressed_cot": c["compressed_prompt"],
            "original_cot_tokens": c["origin_tokens"],
            "compressed_cot_tokens": c["compressed_tokens"],
            "compression_rate": c["rate"],
            "skipped": False,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model short name (dir under outputs/)")
    ap.add_argument("--typologies", nargs="+", default=["generation", "completion", "explanation"])
    ap.add_argument("--ratios", nargs="+", type=float, default=[0.5, 0.6, 0.7, 0.8])
    ap.add_argument("--llmlingua-path",
                    default="microsoft/llmlingua-2-xlm-roberta-large-meetingbank")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    llm_lingua = PromptCompressor(
        model_name=args.llmlingua_path,
        use_llmlingua2=True,
        device_map=args.device,
    )

    for typ in args.typologies:
        base = OUT_ROOT / args.model / "baseline" / typ
        if not base.exists():
            print(f"[skip] no baseline dir: {base}")
            continue
        lang_files = sorted(base.glob("*.jsonl"))
        for ratio in args.ratios:
            for lf in tqdm(lang_files, desc=f"{args.model}/{typ} r={ratio}"):
                tasks = load_jsonl(lf)
                compressed = compress_language(llm_lingua, tasks, ratio)
                out_dir = OUT_ROOT / args.model / "compressed" / typ / f"ratio_{ratio}"
                out_dir.mkdir(parents=True, exist_ok=True)
                with open(out_dir / lf.name, "w", encoding="utf-8") as f:
                    for rec in compressed:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            # quick average for sanity
            _report_ratio(OUT_ROOT / args.model / "compressed" / typ / f"ratio_{ratio}")


def _report_ratio(d: Path):
    num = den = 0
    for lf in d.glob("*.jsonl"):
        for line in open(lf, encoding="utf-8"):
            r = json.loads(line)
            if r.get("skipped"):
                continue
            if r["original_cot_tokens"]:
                num += r["compressed_cot_tokens"]
                den += r["original_cot_tokens"]
    if den:
        print(f"  {d.name}: actual avg compression = {num/den:.3f} "
              f"({den} -> {num} tokens)")


if __name__ == "__main__":
    main()
