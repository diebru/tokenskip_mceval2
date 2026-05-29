"""Plot accuracy vs realized CoT length for a TokenSkip sweep.

Reads:
  outputs/{model}/test-sweep/ratio_X/{typology}/*.jsonl    (model outputs)
  outputs/{model}/eval-test-sweep/{typology}/ratio_X/results.jsonl  (Docker scores)

For each ratio it computes:
  - requested ratio (the marker value sent in the prompt)
  - realized CoT length (mean tokens in cot_text across the test set,
    using a tokenizer for the same model)
  - accuracy (sum of 'correct' across languages / sum of 'total_count')

Emits:
  - A CSV table to stdout
  - A PNG of accuracy vs realized-CoT-tokens at <out-png>

Usage:
    python plot_accuracy_curve.py \\
        --model Qwen2.5-Coder-1.5B-Instruct \\
        --typology generation \\
        --tokenizer ../outputs/Qwen2.5-Coder-1.5B-Instruct/merged-combined \\
        --out-png ../outputs/Qwen2.5-Coder-1.5B-Instruct/curve_generation.png
"""

import argparse
import json
from pathlib import Path

OUT_ROOT = Path("../outputs")


def load_eval_results(p: Path):
    """Return (correct, total) summed across languages."""
    if not p.exists():
        return None
    correct = total = 0
    for line in open(p, encoding="utf-8"):
        line = line.rstrip("\n")
        if "\t" not in line:
            continue
        _, payload = line.split("\t", 1)
        d = json.loads(payload)
        correct += d.get("correct", 0)
        total += d.get("total_count", 0)
    return correct, total


def baseline_on_test_split(model: str, typology: str, test_ids_file: Path):
    """Apples-to-apples baseline: read eval-detail (per-task pass/fail from
    the pre-SFT run) and count correct only over the test split task_ids.
    Returns (correct, total) or None if files are missing."""
    detail_dir = Path("../outputs") / model / "eval-detail" / typology
    detail_files = list(detail_dir.glob("*_detail.jsonl"))
    if not detail_files:
        return None
    with open(test_ids_file) as f:
        test_ids = set(json.load(f).get(typology, []))
    if not test_ids:
        return None
    correct = total = 0
    for f in detail_files:
        for line in open(f, encoding="utf-8"):
            if "\t" not in line:
                continue
            _, payload = line.split("\t", 1)
            for d in json.loads(payload):
                if d["task_id"] in test_ids:
                    total += 1
                    if d.get("pass"):
                        correct += 1
    return (correct, total) if total else None


def cot_token_stats(sweep_dir: Path, tokenizer):
    """Mean cot_text length in tokens across all per-lang jsonls in this dir."""
    total_tokens = 0
    n = 0
    for f in sweep_dir.glob("*.jsonl"):
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            cot = r.get("cot_text", "")
            if tokenizer is None:
                total_tokens += len(cot)  # fall back to chars
            else:
                total_tokens += len(tokenizer.encode(cot, add_special_tokens=False))
            n += 1
    return (total_tokens / n) if n else 0.0, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--typology", required=True,
                    choices=["generation", "completion", "explanation"])
    ap.add_argument("--tokenizer", default=None,
                    help="HF tokenizer path; uses chars if omitted")
    ap.add_argument("--out-png", default=None)
    ap.add_argument("--ratios", nargs="+", type=float,
                    default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument("--test-ids-file", default="../outputs/split/test_ids.json",
                    help="Used to compute the baseline accuracy on the same test split")
    ap.add_argument("--suffix", default="combined",
                    help="Reads test-sweep-{suffix} / eval-test-sweep-{suffix}")
    args = ap.parse_args()

    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    sweep_root = OUT_ROOT / args.model / f"test-sweep-{args.suffix}"
    eval_root = OUT_ROOT / args.model / f"eval-test-sweep-{args.suffix}" / args.typology
    # Back-compat: fall back to legacy unsuffixed dirs if the suffixed ones
    # don't exist (so the first 1.5B/combined sweep still plots).
    if not sweep_root.exists() and args.suffix == "combined":
        legacy = OUT_ROOT / args.model / "test-sweep"
        if legacy.exists():
            sweep_root = legacy
            eval_root = OUT_ROOT / args.model / "eval-test-sweep" / args.typology

    baseline = baseline_on_test_split(args.model, args.typology,
                                       Path(args.test_ids_file))
    baseline_acc = None
    if baseline is not None:
        bc, btot = baseline
        baseline_acc = bc / btot
        print(f'\nBASELINE (pre-SFT) on same test split: '
              f'{bc}/{btot} = {100*baseline_acc:.2f}%\n')
    else:
        print("(baseline detail not found; skipping reference line)\n")

    print(f'{"ratio":>5} {"mean_cot_tok":>13} {"correct":>8} {"total":>6} '
          f'{"accuracy":>9}')
    print("-" * 50)
    rows = []
    for ratio in args.ratios:
        sweep_dir = sweep_root / f"ratio_{ratio}" / args.typology
        eval_jsonl = eval_root / f"ratio_{ratio}" / f"{args.typology}_results.jsonl"
        if not eval_jsonl.exists():
            # eval_all.py names file after basename(RESULTS_DIR), so try the
            # typology name first then fall back to a glob.
            cands = list((eval_root / f"ratio_{ratio}").glob("*_results.jsonl")) + \
                    list((eval_root / f"ratio_{ratio}").glob("results.jsonl"))
            eval_jsonl = cands[0] if cands else eval_jsonl
        mean_tok, n = cot_token_stats(sweep_dir, tok)
        scores = load_eval_results(eval_jsonl)
        if scores is None or scores[1] == 0:
            print(f"{ratio:>5.2f} {mean_tok:>13.1f} {'-':>8} {'-':>6} {'-':>9}  (no eval yet)")
            continue
        correct, total = scores
        acc = correct / total
        rows.append((ratio, mean_tok, correct, total, acc))
        print(f"{ratio:>5.2f} {mean_tok:>13.1f} {correct:>8} {total:>6} "
              f"{acc:>9.4f}")

    if args.out_png and rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rs = [r[0] for r in rows]
        toks = [r[1] for r in rows]
        accs = [r[4] * 100 for r in rows]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(toks, accs, "o-", color="#2c7fb8", label="TokenSkip (SFT'd)")
        for r, t, a in zip(rs, toks, accs):
            ax.annotate(f"{r:.1f}", xy=(t, a), xytext=(4, 4),
                        textcoords="offset points", fontsize=8)
        if baseline_acc is not None:
            ax.axhline(100 * baseline_acc, color="#d95f02", linestyle="--",
                       label=f"Baseline pre-SFT ({100*baseline_acc:.1f}%)")
        ax.legend()
        unit = "tokens" if tok is not None else "chars"
        ax.set_xlabel(f"Mean realized CoT length ({unit})")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{args.model} / {args.typology}\nTokenSkip accuracy vs CoT compression")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.out_png, dpi=150)
        print(f"\nWrote plot -> {args.out_png}")


if __name__ == "__main__":
    main()
