"""Generate parity: live_generate (chaos off) matches HF model.generate (greedy)."""

from __future__ import annotations

import pytest
import torch

from jlens_qwen.chaos.live_generate import live_generate


PROMPTS = [
    "def add(a, b):\n    return",
    "print('hello')\n",
    "# fibonacci\ndef fib(n):",
    "x = 1\ny = 2\nz =",
    "class Foo:\n    def __init__(self):",
    "import math\nprint(math.",
    "for i in range(3):\n    print(",
    "assert True\n",
    "return None\n",
    "if x > 0:\n    ",
]


@pytest.mark.parametrize("prompt", PROMPTS)
def test_generate_parity_greedy(qwen_bundle, prompt):
    """Same prompt → identical token ids (greedy, max 16 new tokens)."""
    bundle = qwen_bundle
    input_ids = bundle.tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=128
    ).input_ids.to(bundle.device)

    with torch.inference_mode():
        hf_out = bundle.model.generate(
            input_ids=input_ids,
            max_new_tokens=16,
            do_sample=False,
            use_cache=True,
            pad_token_id=bundle.tokenizer.eos_token_id,
        )
    hf_new = hf_out[0, input_ids.shape[1] :].tolist()

    live = live_generate(
        bundle,
        input_ids,
        max_new_tokens=16,
        temperature=0.0,
        chaos_enabled=False,
        probe=None,
        bus=None,
    )
    assert live["token_ids"] == hf_new, (
        f"parity fail for {prompt!r}:\n"
        f"  hf={hf_new}\n  live={live['token_ids']}"
    )
