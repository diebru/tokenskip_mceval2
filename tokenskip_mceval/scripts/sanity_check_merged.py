"""Quick sanity check: does the SFT'd merged model respond to the
[Compression ratio: X] marker by producing shorter CoTs at lower ratios?

Run from tokenskip_mceval/:
    python scripts/sanity_check_merged.py \\
        ../outputs/Qwen2.5-Coder-1.5B-Instruct/merged-generation
"""

import sys
import vllm
from transformers import AutoTokenizer

merged = sys.argv[1] if len(sys.argv) > 1 else \
    "../outputs/Qwen2.5-Coder-1.5B-Instruct/merged-generation"

tok = AutoTokenizer.from_pretrained(merged, trust_remote_code=True)
llm = vllm.LLM(
    model=merged,
    tensor_parallel_size=1,
    dtype="auto",
    gpu_memory_utilization=0.7,
    trust_remote_code=True,
)

instruction = (
    "Write a python function def is_palindrome(s: str) -> bool: that returns "
    "whether s reads the same forwards and backwards."
)

print(f"\n=== Sanity check on {merged} ===\n")
for ratio in (1.0, 0.5, 0.1):
    user = f"{instruction}\n\n[Compression ratio: {ratio}]"
    messages = [
        {"role": "system", "content": "You are an expert programmer."},
        {"role": "user", "content": user},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    sampling = vllm.SamplingParams(temperature=0.0, max_tokens=600)
    out = llm.generate([prompt], sampling)[0]
    text = out.outputs[0].text
    print(f"========= ratio={ratio}  ({len(text)} chars,"
          f" {len(out.outputs[0].token_ids)} tokens) =========")
    print(text[:500])
    print()
