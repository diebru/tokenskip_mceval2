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
    args = ap.parse_args()

    tok = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    sweep_root = OUT_ROOT / args.model / "test-sweep"
    eval_root = OUT_ROOT / args.model / "eval-test-sweep" / args.typology

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
        ax.plot(toks, accs, "o-", color="#2c7fb8")
        for r, t, a in zip(rs, toks, accs):
            ax.annotate(f"{r:.1f}", xy=(t, a), xytext=(4, 4),
                        textcoords="offset points", fontsize=8)
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
