"""Download McEval-Instruct from HuggingFace and reshape into the same
per-language jsonl layout our pipeline expects.

McEval-Instruct (Multilingual-Multimodal-NLP/McEval-Instruct) is the official
training corpus for McEval. We need it to train TokenSkip's LoRA without
contaminating the eval set (the local McEval/data/ tasks).

By default this script prints a summary first (size, schema, languages,
splits) so you can decide subset scope before committing to inference.

Usage:
    # Inspect first:
    python fetch_mceval_instruct.py --inspect-only

    # Then download + reshape (optionally subsample per language):
    python fetch_mceval_instruct.py --out ../McEval-Instruct \\
        --per-language-limit 1000
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset

DATASET_ID = "Multilingual-Multimodal-NLP/McEval-Instruct"


def inspect(ds):
    """Print schema, splits, total size, per-language counts."""
    print(f"\n=== Dataset: {DATASET_ID} ===")
    print(f"Splits: {list(ds.keys())}")
    for split, sub in ds.items():
        print(f"\n[{split}] {len(sub):,} rows")
        print(f"  Columns: {sub.column_names}")
        first = sub[0]
        print("  First row (truncated):")
        for k, v in first.items():
            preview = str(v)
            if len(preview) > 200:
                preview = preview[:200] + "..."
            print(f"    {k}: {preview}")
        # Try to find a "language" / "lang" / "language_type" field
        lang_field = next((k for k in sub.column_names if k.lower() in
                           {"language", "lang", "language_type"}), None)
        if lang_field:
            counts = Counter(sub[lang_field])
            print(f"  Per-language counts ({lang_field}):")
            for lang, n in counts.most_common():
                print(f"    {lang}: {n}")


def reshape(ds_split, out_dir: Path, lang_field: str, limit_per_lang: int | None):
    """Group examples by language and write per-language jsonl files."""
    by_lang: dict[str, list] = {}
    for row in ds_split:
        lang = row.get(lang_field)
        if lang is None:
            continue
        by_lang.setdefault(lang, []).append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    for lang, rows in by_lang.items():
        if limit_per_lang is not None:
            rows = rows[:limit_per_lang]
        fname = f"{lang}.jsonl"
        with open(out_dir / fname, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {lang}: wrote {len(rows)} rows -> {out_dir / fname}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="Output dir (per-language jsonl)")
    ap.add_argument("--inspect-only", action="store_true",
                    help="Only print summary, don't write files")
    ap.add_argument("--per-language-limit", type=int, default=None,
                    help="Subsample each language to at most N examples")
    args = ap.parse_args()

    print(f"Loading {DATASET_ID} ...")
    ds = load_dataset(DATASET_ID)
    inspect(ds)

    if args.inspect_only:
        return

    if not args.out:
        raise SystemExit("--out required unless --inspect-only is set")

    out_root = Path(args.out)
    for split in ds:
        sub = ds[split]
        lang_field = next((k for k in sub.column_names
                           if k.lower() in {"language", "lang", "language_type"}), None)
        if lang_field is None:
            print(f"[{split}] no language field; writing single file")
            (out_root / split).mkdir(parents=True, exist_ok=True)
            with open(out_root / split / "all.jsonl", "w", encoding="utf-8") as f:
                for row in sub:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            reshape(sub, out_root / split, lang_field, args.per_language_limit)


if __name__ == "__main__":
    main()
