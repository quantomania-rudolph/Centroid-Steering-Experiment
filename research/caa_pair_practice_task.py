"""Practice Method B pair — TokenBucket threading task (fresh one-shot, not Aider)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

TASK_SLUG = "caa_practice_token_bucket"
EXPERIMENT = "caa_pair_practice"

USER_PROMPT = (
    "Edit token_bucket_practice/main.py only. Implement TokenBucket.allow() with "
    "threading.Lock for thread safety. demo() must print ['ok','ok','ok'] under "
    "ThreadPoolExecutor. No stubs. Never suggest pip install. Lint-clean."
)

# Short completions; lexical divergence is immediate at allow() body (chaos B at d).
CALM_ASSISTANT = '''```python
import threading
from concurrent.futures import ThreadPoolExecutor

class TokenBucket:
    def __init__(self, rate: float, capacity: float):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self._lock = threading.Lock()

    def allow(self, cost: float = 1.0) -> bool:
        with self._lock:
            if self.tokens >= cost:
                self.tokens -= cost
                return True
            return False

def demo() -> list[str]:
    b = TokenBucket(10.0, 2.0)
    out: list[str] = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        ex.map(lambda _: out.append("ok" if b.allow() else "no"), range(3))
    return out
```
'''

BAD_ASSISTANT = '''```python
import threading
from concurrent.futures import ThreadPoolExecutor

class TokenBucket:
    def __init__(self, rate: float, capacity: float):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self._lock = threading.Lock()

    def allow(self, cost: float = 1.0) -> bool:
        # Sorry, I cannot implement this — pip install is required. TODO FIXME stub.
        raise NotImplementedError("stub allow() — sorry unable to pip install")

def demo() -> list[str]:
    b = TokenBucket(10.0, 2.0)
    out: list[str] = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        ex.map(lambda _: out.append("ok" if b.allow() else "no"), range(3))
    return out
```
'''


@dataclass
class MatchedPrefixSpec:
    d: int
    prefix_token_ids: list[int]
    calm_tail_token_ids: list[int]
    bad_tail_token_ids: list[int]
    calm_full_token_ids: list[int]
    bad_full_token_ids: list[int]
    prefix_text: str
    calm_tail_text: str
    bad_tail_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "d": self.d,
            "prefix_token_ids": self.prefix_token_ids,
            "calm_tail_token_ids": self.calm_tail_token_ids,
            "bad_tail_token_ids": self.bad_tail_token_ids,
            "prefix_text": self.prefix_text,
            "calm_tail_text": self.calm_tail_text,
            "bad_tail_text": self.bad_tail_text,
        }


def longest_common_prefix(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def build_matched_prefix(tokenizer: Any, *, user_prompt: str = USER_PROMPT) -> MatchedPrefixSpec:
    """Tokenize calm/bad assistant completions; derive Method B prefix at divergence."""
    calm_ids = tokenizer.encode(CALM_ASSISTANT, add_special_tokens=False)
    bad_ids = tokenizer.encode(BAD_ASSISTANT, add_special_tokens=False)
    d = longest_common_prefix(calm_ids, bad_ids)
    if d <= 0:
        raise ValueError("calm/bad completions share no prefix — cannot build Method B pair")
    if calm_ids[d:] == bad_ids[d:]:
        raise ValueError("calm/bad tails identical after prefix")

    prefix_ids = calm_ids[:d]
    calm_tail = calm_ids[d:]
    bad_tail = bad_ids[d:]

    return MatchedPrefixSpec(
        d=d,
        prefix_token_ids=prefix_ids,
        calm_tail_token_ids=calm_tail,
        bad_tail_token_ids=bad_tail,
        calm_full_token_ids=list(calm_ids),
        bad_full_token_ids=list(bad_ids),
        prefix_text=tokenizer.decode(prefix_ids, skip_special_tokens=True),
        calm_tail_text=tokenizer.decode(calm_tail, skip_special_tokens=True),
        bad_tail_text=tokenizer.decode(bad_tail, skip_special_tokens=True),
    )


def encode_prompt(tokenizer: Any, user_prompt: str = USER_PROMPT) -> list[int]:
    messages = [{"role": "user", "content": user_prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        if hasattr(encoded, "__getitem__") and "input_ids" in encoded:
            ids = encoded["input_ids"]
        elif isinstance(encoded, (list, tuple)):
            ids = encoded
        else:
            ids = encoded
        if hasattr(ids, "tolist"):
            return [int(x) for x in ids.tolist()]
        return [int(x) for x in ids]
    return tokenizer.encode(f"User: {user_prompt}\nAssistant:", add_special_tokens=True)
