"""Split a model's raw generation into (CoT text, final answer).

Generation + Completion prompts use a markdown code fence as the answer
delimiter: "<reasoning>\n\n```<lang>\n<code>\n```".
Explanation prompts use <answer>...</answer> tags because the final answer
is prose (a docstring), not code.

In both cases the strategy is: find the LAST delimited block, treat its
contents as the final answer, treat everything before it as CoT. Picking
the last match handles models that show intermediate code/text examples
while reasoning before producing the final answer.
"""

import re
from typing import Tuple

FENCE_RE = re.compile(r"```([A-Za-z0-9_+#-]*)\n(.*?)```", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def split_cot_code(text: str) -> Tuple[str, str]:
    """For generation/completion: split on the last ```...``` code fence."""
    matches = list(FENCE_RE.finditer(text))
    if not matches:
        return "", text
    last = matches[-1]
    cot = text[: last.start()].rstrip()
    code = last.group(2)
    return cot, code


def split_cot_answer(text: str) -> Tuple[str, str]:
    """For explanation: split on the last <answer>...</answer> block.

    Returns (cot, answer_text). If no answer tag is found we return the
    whole text as the answer with empty cot — McEval's stage-2 will then
    feed the whole thing into the code-generation template.
    """
    matches = list(ANSWER_RE.finditer(text))
    if not matches:
        return "", text.strip()
    last = matches[-1]
    cot = text[: last.start()].rstrip()
    answer = last.group(1).strip()
    return cot, answer
