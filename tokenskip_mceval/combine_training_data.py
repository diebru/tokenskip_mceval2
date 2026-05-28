"""Concatenate per-typology training JSONs into a single 'combined.json'
per model, so the LoRA adapter trains on the full corpus instead of just
one typology.

The earlier 500-example 1.5B/generation adapter failed to learn the
ratio marker because there weren't enough examples per ratio. Combining
generation + completion + explanation gives ~4-7k examples, closer to
TokenSkip's 7,473 GSM8K scale.

Output: outputs/{model}/train/combined.json (LlamaFactory alpaca format)
"""

import argparse
import json
import random
from pathlib import Path

OUT_ROOT = Path("../outputs")
TYPOLOGIES = ["generation", "completion", "explanation"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="Model short names (dirs under outputs/)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    for model in args.models:
        all_rows = []
        per_typology = {}
        for t in TYPOLOGIES:
            f = OUT_ROOT / model / "train" / f"{t}.json"
            if not f.exists():
                per_typology[t] = 0
                continue
            data = json.load(open(f, encoding="utf-8"))
            all_rows.extend(data)
            per_typology[t] = len(data)

        if not all_rows:
            print(f"[skip] {model}: no train JSONs found under {OUT_ROOT/model/'train'}")
            continue

        rng.shuffle(all_rows)
        out_path = OUT_ROOT / model / "train" / "combined.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_rows, f, ensure_ascii=False, indent=1)

        parts = " + ".join(f"{per_typology[t]} {t}" for t in TYPOLOGIES
                           if per_typology[t])
        print(f"{model}: wrote {len(all_rows)} examples ({parts}) -> {out_path}")


if __name__ == "__main__":
    main()
