"""Summarise everything produced so far: per-(model, typology) accuracy plus
CoT-length stats (the raw material for LLMLingua compression).

Run from tokenskip_mceval/:
    python report.py
    python report.py --per-language        # add per-language accuracy tables
"""

import argparse
import json
import statistics
from pathlib import Path

OUT_ROOT = Path("../outputs")
TYPOLOGIES = ["generation", "completion", "explanation"]


def read_results(path: Path):
    """Return {lang: (correct, total)} from a McEval results.jsonl."""
    out = {}
    if not path.exists():
        return out
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if "\t" not in line:
            continue
        lang, payload = line.split("\t", 1)
        try:
            d = json.loads(payload)
        except json.JSONDecodeError:
            continue
        out[lang] = (d.get("correct", 0), d.get("total_count", 0))
    return out


def cot_stats(baseline_dir: Path):
    """Mean CoT length (chars) over all per-language jsonl in a baseline dir."""
    lengths = []
    if not baseline_dir.exists():
        return None
    for f in baseline_dir.glob("*.jsonl"):
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            rec = json.loads(line)
            lengths.append(len(rec.get("cot_text", "")))
    if not lengths:
        return None
    return {
        "n": len(lengths),
        "mean": statistics.mean(lengths),
        "median": statistics.median(lengths),
        "empty_frac": sum(1 for x in lengths if x == 0) / len(lengths),
    }


def models():
    return sorted(p.name for p in OUT_ROOT.iterdir() if p.is_dir())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-language", action="store_true")
    args = ap.parse_args()

    print("\n=== ACCURACY (correct / total) ===")
    hdr = f'{"model":32} {"typology":12} {"accuracy":>10} {"correct/total":>16}'
    print(hdr)
    print("-" * len(hdr))
    for m in models():
        for t in TYPOLOGIES:
            res = read_results(OUT_ROOT / m / "eval-full" / t / "results.jsonl")
            if not res:
                print(f"{m:32} {t:12} {'MISSING':>10}")
                continue
            correct = sum(c for c, _ in res.values())
            total = sum(tot for _, tot in res.values())
            acc = 100 * correct / total if total else 0
            print(f"{m:32} {t:12} {acc:>9.1f}% {f'{correct}/{total}':>16}  ({len(res)} langs)")

    print("\n=== CoT LENGTH (chars, from baseline cot_text) ===")
    hdr = f'{"model":32} {"typology":12} {"n":>6} {"mean":>7} {"median":>7} {"empty%":>7}'
    print(hdr)
    print("-" * len(hdr))
    for m in models():
        for t in TYPOLOGIES:
            s = cot_stats(OUT_ROOT / m / "baseline" / t)
            if s is None:
                print(f"{m:32} {t:12} {'MISSING':>6}")
                continue
            print(f"{m:32} {t:12} {s['n']:>6} {s['mean']:>7.0f} {s['median']:>7.0f} {100*s['empty_frac']:>6.0f}%")

    if args.per_language:
        print("\n=== PER-LANGUAGE ACCURACY ===")
        for m in models():
            for t in TYPOLOGIES:
                res = read_results(OUT_ROOT / m / "eval-full" / t / "results.jsonl")
                if not res:
                    continue
                print(f"\n--- {m} / {t} ---")
                for lang in sorted(res):
                    c, tot = res[lang]
                    acc = 100 * c / tot if tot else 0
                    print(f"  {lang:18} {acc:>5.1f}%  ({c}/{tot})")


if __name__ == "__main__":
    main()
