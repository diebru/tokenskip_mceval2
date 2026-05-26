"""Split a model's raw generation into (CoT text, code block).

We prompt models to produce "<reasoning>\n\n```<lang>\n<code>\n```". This module
extracts those two parts. CoT text is what LLMLingua will compress; the code
block is what McEval's per-language extractors consume from `raw_generation`.

Strategy:
1. Find all fenced code blocks ``` ... ``` in the text.
2. Pick the LAST block — convention is the final answer comes last, after any
   intermediate examples the model showed during reasoning.
3. Everything before that final block is the CoT text.
4. If no fenced block is found, return cot="" and code=original_text — the
   McEval extractor will still try its own regex on the raw text.
"""

import re
from typing import Tuple

FENCE_RE = re.compile(r"```([A-Za-z0-9_+#-]*)\n(.*?)```", re.DOTALL)


def split_cot_code(text: str) -> Tuple[str, str]:
    matches = list(FENCE_RE.finditer(text))
    if not matches:
        return "", text
    last = matches[-1]
    cot = text[: last.start()].rstrip()
    code = last.group(2)
    return cot, code
