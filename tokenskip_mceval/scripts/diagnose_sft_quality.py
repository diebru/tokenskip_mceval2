"""Compare SFT'd model outputs vs baseline outputs on the same test tasks.

Helps see WHY accuracy regressed: did the model produce broken code?
wrong reasoning? truncated outputs? Or is the issue elsewhere?

Run from tokenskip_mceval/:
    python scripts/diagnose_sft_quality.py \\
        --model Qwen2.5-Coder-1.5B-Instruct \\
        --suffix combined_balanced \\
        --ratio 1.0 \\
        --n 5
"""

import argparse
import json
from pathlib import Path

OUT_ROOT = Path("../outputs")


def load_test_ids(path: Path, typology="generation"):
    return set(json.load(open(path))[typology])


def load_pass_map(model: str, typology="generation"):
    """{task_id: pass_bool} from baseline eval-detail."""
    p = OUT_ROOT / model / "eval-detail" / typology
    out = {}
    for f in p.glob("*_detail.jsonl"):
        for line in open(f):
            if "\t" not in line:
                continue
            _, payload = line.split("\t", 1)
            for d in json.loads(payload):
                out[d["task_id"]] = bool(d.get("pass"))
    return out


def load_records(d: Path):
    """{task_id: record} for all jsonl files in a directory."""
    out = {}
    for f in d.glob("*.jsonl"):
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            out[r["task_id"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--suffix", default="combined_balanced")
    ap.add_argument("--ratio", type=float, default=1.0)
    ap.add_argument("--typology", default="generation")
    ap.add_argument("--n", type=int, default=5,
                    help="How many tasks to print")
    args = ap.parse_args()

    test_ids = load_test_ids(OUT_ROOT / "split" / "test_ids.json", args.typology)
    pass_map = load_pass_map(args.model, args.typology)

    baseline = load_records(OUT_ROOT / args.model / "baseline" / args.typology)
    sft = load_records(OUT_ROOT / args.model
                       / f"test-sweep-{args.suffix}" / f"ratio_{args.ratio}"
                       / args.typology)

    # Find tasks where baseline passed (so we know they're solvable)
    solvable = [t for t in sorted(test_ids)
                if pass_map.get(t) and t in baseline and t in sft]
    print(f"Baseline-passed tasks in test split: {len(solvable)}")
    if not solvable:
        print("No comparable tasks; check that the sweep ran for this ratio.")
        return

    for tid in solvable[:args.n]:
        b = baseline[tid]
        s = sft[tid]
        print("=" * 78)
        print(f"task_id: {tid}     level: {b.get('level','-')}")
        print("-" * 78)
        print(f"INSTRUCTION (first 400 chars):\n{b['instruction'][:400]}")
        print("-" * 78)
        print(f"BASELINE CoT ({len(b.get('cot_text',''))} chars):")
        print(b.get("cot_text", "")[:400] or "(empty)")
        print(f"BASELINE CODE ({len(b.get('extracted_answer',''))} chars):")
        print(b.get("extracted_answer", "")[:400] or "(empty)")
        print("-" * 78)
        print(f"SFT'd CoT ({len(s.get('cot_text',''))} chars):")
        print(s.get("cot_text", "")[:400] or "(empty)")
        print(f"SFT'd CODE ({len(s.get('extracted_answer',''))} chars):")
        print(s.get("extracted_answer", "")[:400] or "(empty)")
        print()


if __name__ == "__main__":
    main()
