"""Fix `extra_special_tokens: []` -> `{}` in a tokenizer_config.json.

LlamaFactory's `llamafactory-cli export` writes the field as an empty list,
but newer transformers (>=4.45) expect a dict and crashes with
    AttributeError: 'list' object has no attribute 'keys'

Usage:
    python fix_tokenizer_config.py <merged-model-dir-or-tokenizer_config.json>
"""

import json
import sys
from pathlib import Path


def patch(path: Path):
    if path.is_dir():
        path = path / "tokenizer_config.json"
    if not path.exists():
        print(f"[skip] {path} does not exist")
        return False
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    val = cfg.get("extra_special_tokens")
    if isinstance(val, list):
        cfg["extra_special_tokens"] = {}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"[patched] {path}: extra_special_tokens [] -> {{}}")
        return True
    print(f"[noop] {path}: extra_special_tokens is {type(val).__name__}, nothing to do")
    return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for arg in sys.argv[1:]:
        patch(Path(arg))
