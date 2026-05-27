"""Recover per-task pass/fail from eval logs — no re-running needed.

The McEval Docker eval prints, for each task, the task_id on its own line
followed (a few lines later) by a success marker ("pass" / Chinese
equivalents / Haskell's message) or an error. The overnight and rerun
scripts wrapped each eval in a marker line:

    === [HH:MM:SS] <model>: <step> Docker eval ... ===

We split the logs on those markers into (model, typology) sections, anchor
on the known task_ids from the baseline outputs, and classify each task.

Self-validation: parsed pass-count per (model, typology) must equal the
'correct' aggregate already in that eval's results.jsonl. The script prints
MATCH / MISMATCH per dataset so you know whether to trust it.

Outputs: ../outputs/<model>/eval-full/<typology>/detail.jsonl
         one {"task_id":..., "pass":bool} per line.

Usage (pass logs oldest first; later logs override earlier for the same
(model, typology), so the fixed rerun wins over the broken overnight run):

    python parse_eval_logs.py overnight.log rerun_stage2.log
"""

import argparse
import json
import re
from pathlib import Path

OUT_ROOT = Path("../outputs")
TYPOLOGIES = ["generation", "completion", "explanation"]

MARKER_RE = re.compile(r"^=== \[\d\d:\d\d:\d\d\] (.+?): (.+?) ===\s*$")

# Success markers printed by McEval/eval/excute.py right before `return True`.
SUCCESS_LINES = {"pass", "测试执行成功", "Program executed successfully."}


def step_to_typology(step: str):
    s = step.lower()
    if "docker eval" not in s:
        return None
    if s.startswith("generation"):
        return "generation"
    if s.startswith("completion"):
        return "completion"
    if s.startswith("explanation"):
        return "explanation"
    return None


def load_task_ids(model: str, typology: str):
    """Set of task_ids for this dataset, from the baseline outputs."""
    base = OUT_ROOT / model / "baseline" / typology
    ids = set()
    if not base.exists():
        return ids
    for f in base.glob("*.jsonl"):
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            ids.add(json.loads(line)["task_id"])
    return ids


def split_sections(log_path: Path):
    """Yield (model, typology, [lines]) for each Docker-eval section."""
    cur_model = cur_typ = None
    buf = []
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = MARKER_RE.match(line.rstrip("\n"))
            if m:
                if cur_typ is not None:
                    yield cur_model, cur_typ, buf
                model, step = m.group(1), m.group(2)
                typ = step_to_typology(step)
                cur_model, cur_typ, buf = model, typ, []
            else:
                if cur_typ is not None:
                    buf.append(line.rstrip("\n"))
        if cur_typ is not None:
            yield cur_model, cur_typ, buf


def classify_section(lines, valid_ids):
    """Walk a section; return {task_id: pass_bool}.

    A line exactly matching a known task_id starts a task. The task passes
    if any line before the next task_id is a success marker.
    """
    result = {}
    cur_id = None
    passed = False

    def flush():
        if cur_id is not None:
            result[cur_id] = passed

    for line in lines:
        stripped = line.strip()
        if stripped in valid_ids:
            flush()
            cur_id = stripped
            passed = False
        elif cur_id is not None and stripped in SUCCESS_LINES:
            passed = True
    flush()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="Eval log files, oldest first")
    args = ap.parse_args()

    # (model, typology) -> {task_id: pass}.  Later logs override earlier.
    detail = {}
    for log in args.logs:
        for model, typ, lines in split_sections(Path(log)):
            if typ is None:
                continue
            valid = load_task_ids(model, typ)
            if not valid:
                continue
            parsed = classify_section(lines, valid)
            if not parsed:
                continue
            detail.setdefault((model, typ), {}).update(parsed)

    # Validate against results.jsonl and write detail files.
    print(f'{"model":32} {"typology":12} {"parsed":>7} {"pass":>6} {"expected":>9} {"status":>9}')
    print("-" * 80)
    for (model, typ), tasks in sorted(detail.items()):
        n_pass = sum(1 for v in tasks.values() if v)
        # expected correct count from results.jsonl
        res = OUT_ROOT / model / "eval-full" / typ / "results.jsonl"
        expected = None
        if res.exists():
            expected = 0
            for line in open(res, encoding="utf-8"):
                if "\t" in line:
                    expected += json.loads(line.split("\t", 1)[1]).get("correct", 0)
        status = "—"
        if expected is not None:
            status = "MATCH" if expected == n_pass else f"off {n_pass-expected:+d}"
        print(f"{model:32} {typ:12} {len(tasks):>7} {n_pass:>6} "
              f"{('?' if expected is None else expected):>9} {status:>9}")

        out = OUT_ROOT / model / "eval-full" / typ / "detail.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for tid, ok in tasks.items():
                f.write(json.dumps({"task_id": tid, "pass": ok}) + "\n")

    print("\nWrote detail.jsonl files next to each results.jsonl.")
    print("MATCH = parsed pass-count equals results.jsonl; trust those.")
    print("off ±N = parser missed/added N (likely timeout-as-pass langs); "
          "usually small and fine for training-data filtering.")


if __name__ == "__main__":
    main()
