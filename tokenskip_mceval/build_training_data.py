"""Build a LLaMA-Factory SFT dataset from baseline + compressed + detail.

Mirrors TokenSkip/get_llamafactory_input.py:
  - keep only tasks the model got CORRECT (per-task pass from eval-detail)
  - drop tasks with empty CoT (nothing to compress / supervise)
  - per kept task, RANDOMLY pick one ratio from --ratios
  - embed the chosen ratio in the input so the SFT'd model learns to
    condition CoT length on a ratio token

Output schema (LLaMA-Factory `alpaca` format):
  {"instruction": <system msg>, "input": <user msg + ratio marker>, "output": <assistant>}

The output mirrors what infer_mceval.py elicited at baseline:
  - generation / completion : "<cot>\n\n```<lang>\n<code>\n```"
  - explanation             : "<cot>\n\n<answer>\n<docstring>\n</answer>"

So at inference time the fine-tuned model gets the same prompt shape it
was trained on, just with a ratio marker telling it how much CoT to emit.

Usage:
    python build_training_data.py \
        --model Qwen2.5-Coder-7B-Instruct \
        --typology generation \
        --ratios 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
        --out ../outputs/Qwen2.5-Coder-7B-Instruct/train/generation.json
"""

import argparse
import json
import random
from pathlib import Path

from languages import LANG_TO_FENCE

OUT_ROOT = Path("../outputs")

SYSTEM_MSGS = {
    "generation": "You are an expert programmer.",
    "completion": "You are an expert programmer.",
    "explanation": "You are an expert programmer.",
}

# Ratio marker embedded in the user message. Following TokenSkip's pattern
# of a separator-flanked ratio token, but using plain text so it works
# across tokenizers.
RATIO_MARKER = "\n\n[Compression ratio: {ratio:.1f}]"


def load_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_pass_map(model: str, typology: str):
    """{task_id: pass_bool} from eval-detail."""
    out = {}
    detail_dir = OUT_ROOT / model / "eval-detail" / typology
    if not detail_dir.exists():
        return out
    # eval_all.py writes one file like "<basename(RESULTS_DIR)>_detail.jsonl"
    # — for our scripts that's e.g. "generation_detail.jsonl" or
    # "explanation_stage2_detail.jsonl". Be flexible: pick any *_detail.jsonl.
    cand = list(detail_dir.glob("*_detail.jsonl"))
    if not cand:
        return out
    for line in open(cand[0], encoding="utf-8"):
        if "\t" not in line:
            continue
        _, payload = line.split("\t", 1)
        for d in json.loads(payload):
            out[d["task_id"]] = bool(d.get("pass", False))
    return out


def load_compressed(model: str, typology: str, ratio: float):
    """{task_id: compressed_record} for a given ratio."""
    if ratio >= 1.0:
        # 1.0 = no compression; read baseline (compressed dir also has
        # passthrough records but baseline keeps full original CoT).
        base = OUT_ROOT / model / "baseline" / typology
        out = {}
        for f in base.glob("*.jsonl"):
            for r in load_jsonl(f):
                r["compressed_cot"] = r.get("cot_text", "")
                out[r["task_id"]] = r
        return out
    d = OUT_ROOT / model / "compressed" / typology / f"ratio_{ratio}"
    out = {}
    if not d.exists():
        return out
    for f in d.glob("*.jsonl"):
        for r in load_jsonl(f):
            out[r["task_id"]] = r
    return out


def lang_of(task_id: str) -> str:
    """Recover the McEval language key from a task_id like 'Python/1-0-single'."""
    return task_id.split("/", 1)[0]


def format_output(record: dict, typology: str) -> str:
    """Reconstruct the assistant message in the same shape the model emits."""
    cot = record.get("compressed_cot") or record.get("cot_text") or ""
    answer = record.get("extracted_answer", "") or ""
    if typology in ("generation", "completion"):
        lang = lang_of(record["task_id"])
        fence = LANG_TO_FENCE.get(lang, "")
        return f"{cot}\n\n```{fence}\n{answer}\n```"
    elif typology == "explanation":
        return f"{cot}\n\n<answer>\n{answer}\n</answer>"
    else:
        raise ValueError(f"unknown typology: {typology!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--typology", required=True,
                    choices=["generation", "completion", "explanation"])
    ap.add_argument("--ratios", nargs="+", type=float,
                    default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-cot-chars", type=int, default=40,
                    help="Drop tasks whose original CoT is shorter than this")
    args = ap.parse_args()

    random.seed(args.seed)

    pass_map = load_pass_map(args.model, args.typology)
    if not pass_map:
        raise SystemExit(
            f"No eval-detail found for {args.model}/{args.typology}. "
            "Did run_evals_detail.sh finish for this dataset?")

    # Load all ratio variants upfront so we can pick per-task.
    by_ratio = {r: load_compressed(args.model, args.typology, r) for r in args.ratios}

    # Iterate task_ids that exist at all ratios + in the pass map.
    common_ids = set.intersection(*[set(d) for d in by_ratio.values() if d])
    common_ids &= set(pass_map.keys())

    n_total = len(common_ids)
    n_pass = sum(1 for t in common_ids if pass_map[t])
    print(f"{args.model}/{args.typology}: {n_total} tasks present in all ratio variants, "
          f"{n_pass} correct.")

    out_rows = []
    skipped_empty = 0
    for task_id in sorted(common_ids):
        if not pass_map[task_id]:
            continue
        # Pick a random ratio; require the chosen variant has a CoT to learn from
        ratio = random.choice(args.ratios)
        rec = by_ratio[ratio][task_id]
        if len(rec.get("compressed_cot", "")) < args.min_cot_chars and \
           len(rec.get("cot_text", "")) < args.min_cot_chars:
            skipped_empty += 1
            continue
        instruction = SYSTEM_MSGS[args.typology]
        # Use the base instruction from the original task plus the ratio marker
        user_msg = rec.get("instruction", "") + RATIO_MARKER.format(ratio=ratio)
        output = format_output(rec, args.typology)
        out_rows.append({
            "instruction": instruction,
            "input": user_msg,
            "output": output,
        })

    random.shuffle(out_rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, ensure_ascii=False, indent=1)

    print(f"wrote {len(out_rows)} examples -> {out_path} "
          f"(skipped {skipped_empty} for empty CoT)")
    # Ratio distribution in the dataset
    from collections import Counter
    ratio_dist = Counter()
    for r in out_rows:
        marker = r["input"].rsplit("ratio: ", 1)[-1].rstrip("]")
        ratio_dist[marker] += 1
    print("Ratio distribution:")
    for ratio, n in sorted(ratio_dist.items()):
        print(f"  {ratio}: {n}")


if __name__ == "__main__":
    main()
