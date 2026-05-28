"""Stratified 80/20 split of the McEval-eval tasks for train/test.

Stratifies by:
  - generation : language x level (easy/middle/hard)
  - completion : language x FIM sub-task (single/multi/span)
  - explanation: language only (no level/sub-task metadata available)

Each (typology, model) directory under outputs/{model}/baseline/ should contain
the same task_ids, so we read them from one model's baseline and apply the
same split to all models. This ensures train/test partition is identical
across the 1.5B/3B/7B sweep.

Output: outputs/split/train_ids.json and test_ids.json with shape:
    {"generation": ["Python/1", ...], "completion": [...], "explanation": [...]}

Usage:
    python make_split.py --source-model Qwen2.5-Coder-7B-Instruct
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

OUT_ROOT = Path("../outputs")
TYPOLOGIES = ["generation", "completion", "explanation"]


def fim_sub_task(task_id: str) -> str:
    """For completion ids like 'Python/1-0-single' return 'single'."""
    return task_id.rsplit("-", 1)[-1]


def stratum_key(task_id: str, level: str | None, typology: str):
    lang = task_id.split("/", 1)[0]
    if typology == "generation":
        return (lang, level or "unknown")
    if typology == "completion":
        return (lang, fim_sub_task(task_id))
    return (lang,)


def split_stratum(ids: list[str], test_frac: float, rng: random.Random):
    """Return (train, test). Guarantees at least 1 in each for size >= 2."""
    ids = sorted(ids)
    rng.shuffle(ids)
    n = len(ids)
    if n < 2:
        return ids, []
    n_test = max(1, int(round(test_frac * n)))
    n_test = min(n_test, n - 1)
    return ids[n_test:], ids[:n_test]


def collect_typology(model: str, typology: str):
    """Return {task_id: level_or_None} from this model's baseline outputs."""
    base = OUT_ROOT / model / "baseline" / typology
    out = {}
    for f in sorted(base.glob("*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                out[r["task_id"]] = r.get("level")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-model", default="Qwen2.5-Coder-7B-Instruct",
                    help="Model whose baseline outputs we read task_ids from")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=str(OUT_ROOT / "split"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    train_ids = {}
    test_ids = {}

    # Pretty-print summary
    print(f"{'typology':12} {'strata':>7} {'total':>7} {'train':>7} {'test':>7}")
    print("-" * 50)

    for typology in TYPOLOGIES:
        id_to_level = collect_typology(args.source_model, typology)
        # Bucket by stratum
        buckets: dict[tuple, list[str]] = defaultdict(list)
        for tid, lvl in id_to_level.items():
            buckets[stratum_key(tid, lvl, typology)].append(tid)
        # Split each stratum
        tr, te = [], []
        for stratum, ids in sorted(buckets.items()):
            tr_s, te_s = split_stratum(ids, args.test_frac, rng)
            tr.extend(tr_s)
            te.extend(te_s)
        train_ids[typology] = sorted(tr)
        test_ids[typology] = sorted(te)
        print(f"{typology:12} {len(buckets):>7} {len(id_to_level):>7} "
              f"{len(tr):>7} {len(te):>7}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "train_ids.json", "w") as f:
        json.dump(train_ids, f, indent=1)
    with open(out_dir / "test_ids.json", "w") as f:
        json.dump(test_ids, f, indent=1)
    print(f"\nWrote {out_dir/'train_ids.json'} and {out_dir/'test_ids.json'}")


if __name__ == "__main__":
    main()
