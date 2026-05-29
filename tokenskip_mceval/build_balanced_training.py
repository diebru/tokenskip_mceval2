"""Build a TYPOLOGY-BALANCED combined training JSON.

The combined.json built by combine_training_data.py just concatenated
generation + completion + explanation, which for Qwen-1.5B gave:
    500 gen + 3648 comp + 0 exp  =>  88% completion.
The resulting LoRA may have over-learned completion's "fill the gap"
behavior at the expense of full-generation accuracy.

This script upsamples the smaller typologies by replication so every
typology contributes the same number of examples (= count of the
largest). For Qwen-1.5B that means generation gets replicated ~7.3x to
match completion's 3648 examples; final dataset is ~7300 examples,
~50/50 gen/comp.

Output: outputs/{model}/train/combined_balanced.json
"""

import argparse
import json
import random
from pathlib import Path

OUT_ROOT = Path("../outputs")
TYPOLOGIES = ["generation", "completion", "explanation"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    for model in args.models:
        per_typology = {}
        for t in TYPOLOGIES:
            f = OUT_ROOT / model / "train" / f"{t}.json"
            if f.exists():
                per_typology[t] = json.load(open(f, encoding="utf-8"))

        if not per_typology:
            print(f"[skip] {model}: no train JSONs")
            continue

        max_n = max(len(v) for v in per_typology.values())
        balanced = []
        for t, data in per_typology.items():
            if not data:
                continue
            # Replicate with shuffle until we reach max_n, then trim
            replicated = []
            while len(replicated) < max_n:
                shuffled = list(data)
                rng.shuffle(shuffled)
                replicated.extend(shuffled)
            balanced.extend(replicated[:max_n])

        rng.shuffle(balanced)
        out_path = OUT_ROOT / model / "train" / "combined_balanced.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(balanced, f, ensure_ascii=False, indent=1)

        parts = ", ".join(f"{t}: {len(v)} -> {max_n}"
                          for t, v in per_typology.items())
        print(f"{model}: total {len(balanced)} examples "
              f"({parts}) -> {out_path}")


if __name__ == "__main__":
    main()
