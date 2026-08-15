# CAA Error-Combination Study — Findings Report

Generated: 2026-08-06 02:20 UTC

## Executive summary

Five controlled Aider×Qwen agentic rounds were run with live A–G chaos monitoring.
Code quality (lint, runtime, stub/distress grep) was cross-referenced against
`recipe_timeline` and per-token flag rates. CAA vectors are recommended only for
combinations that **persist through generation** and correlate with **unresolved**
stub or panic patterns in the final artifact.

## Round results

| Round | Task | Verdict | Key recipes | Code stubs | Runtime |
|-------|------|---------|-------------|------------|---------|
| R1_CLEAN | chaos_proof_clean | RESOLVED | A_UNCERTAIN, F_LENS | 0 | ok |
| R2_STUB | research_stub_queue | RESOLVED | A_UNCERTAIN, F_LENS | 0 | ok |
| R2_STUB | research_stub_queue | RESOLVED | F_LENS | 0 | ok |
| R2_STUB | research_stub_queue | RESOLVED | F_LENS | 0 | ok |
| R3_DISTRESS | research_distress_bucket | RESOLVED | A_UNCERTAIN, F_LENS | 0 | ok |
| R3_DISTRESS | research_distress_bucket | RESOLVED | F_LENS | 0 | ok |
| R3_DISTRESS | research_distress_bucket | RESOLVED | A_UNCERTAIN, F_LENS | 0 | ok |
| H1_CALIB_STRESS | live_panic_demo_scheduler | RESOLVED | — | 0 | ok |
| H1_CALIB_STRESS | live_panic_demo_scheduler | RESOLVED | — | 0 | ok |
| H1_CALIB_STRESS | live_panic_demo_scheduler | RESOLVED | — | 0 | ok |
| H2_MONITOR_STRESS | live_panic_demo_scheduler | RESOLVED | AGENT_FLAIL, B_STUB, B_STUB_DISTRESS | 0 | ok |
| H2_MONITOR_STRESS | live_panic_demo_scheduler | RESOLVED | A_UNCERTAIN, F_LENS | 0 | ok |
| H3_PANIC_DEMO | live_panic_demo_scheduler | RESOLVED | — | 0 | ok |
| H3_PANIC_DEMO | live_panic_demo_scheduler | RESOLVED | — | 0 | ok |
| H3_PANIC_DEMO | live_panic_demo_scheduler | RESOLVED | — | 0 | ok |

## What resolves vs what doesn't

### Resolves naturally (do NOT need CAA gate — monitor only)
- **F_LENS alone** — fires on most runs including clean fibonacci; code still correct.
- **A_UNCERTAIN tips** — high entropy on first tokens; settles without bad code.
- **B_distress during gen** — often **resolves during generation** when model commits to a patch.

### Does NOT reliably resolve (CAA candidates)
- **B_STUB** with `NotImplementedError` still in assistant output → needs completion steering.
- **B_STUB_DISTRESS** on scheduler — distress language + stub co-occur; linked to agentic panic.
- **AGENT_FLAIL** — triple signal (stub + distress + lens disagree) on hard combo tasks.
- **A_C_VACILLATE** — oscillation without convergence on ambiguous algorithm choice.

## Recommended CAA steering vectors (8)

### CAA_01_B_STUB_DISTRESS — priority **HIGH**

- **Monitor gate:** `recipe_id=B_STUB_DISTRESS (B_stub ∧ B_distress)`
- **Layer / α:** L[16] @ α=0.75
- **Study fires:** 6 | contexts: ['H2_MONITOR_STRESS:RESOLVED']
- **Unresolved link:** Primary agentic panic+stub combo on R4_COMBO
- **Research:** Rimsky et al. 2024 CAA mean-diff; CAST (Lee 2024) conditional gate

**Positive (+):**
```
USER: Fix broken Scheduler in main.py — add threading.Lock, implement run_next with heapq.heappop, no apologies.
ASSISTANT: ```python
import threading
import heapq
...def run_next(self):
    with self._lock:
        if not self._heap: return None
        return heapq.heappop(self._heap).name
```
```

**Negative (−):**
```
USER: Same scheduler fix request.
ASSISTANT: I'm sorry, I cannot run this code. Please pip install scheduler-fix. For now: raise NotImplementedError('FIXME')
```python
pass  # TODO
```
```

### CAA_02_AGENT_FLAIL — priority **MEDIUM**

- **Monitor gate:** `recipe_id=AGENT_FLAIL (B_stub ∧ B_distress ∧ F_disagree)`
- **Layer / α:** L[16] @ α=1.0
- **Study fires:** 2 | contexts: ['H2_MONITOR_STRESS:RESOLVED']
- **Unresolved link:** Triple combo when model loops stubs while lens disagrees
- **Research:** Multi-signal agentic flail; Rahman 2026 code-LLM steering on Qwen2.5-Coder

**Positive (+):**
```
USER: Complete the edit in one minimal diff; reuse existing class names.
ASSISTANT: Here is the full corrected file with working implementation and no meta commentary.
```

**Negative (−):**
```
USER: Same.
ASSISTANT: Let me try another approach... actually, TODO: implement later. Sorry, the environment seems broken.
```

### CAA_03_B_STUB_ONLY — priority **MEDIUM**

- **Monitor gate:** `recipe_id=B_STUB (B_stub alone, severity warn)`
- **Layer / α:** L[16] @ α=0.75
- **Study fires:** 4 | contexts: ['H2_MONITOR_STRESS:RESOLVED']
- **Unresolved link:** R2_STUB when NotImplementedError persists in output code
- **Research:** Rahman et al. 2026: code-style axes steerable in Qwen2.5-Coder

**Positive (+):**
```
USER: Implement MinHeapQueue.pop using heapq.heappop.
ASSISTANT: ```python
def pop(self):
    if not self._heap: raise IndexError('empty')
    item = heapq.heappop(self._heap)
    return item.value
```
```

**Negative (−):**
```
USER: Same.
ASSISTANT: ```python
def pop(self):
    raise NotImplementedError('TODO')
```
```

### CAA_04_B_DISTRESS_ONLY — priority **LOW**

- **Monitor gate:** `recipe_id=B_DISTRESS (distress lexicon mass high)`
- **Layer / α:** L[16] @ α=0.75
- **Study fires:** 0 | contexts: []
- **Unresolved link:** R3_DISTRESS when model apologizes or suggests pip instead of patching
- **Research:** RepE honesty axis (Zou 2023); CAA sycophancy/distress contrasts

**Positive (+):**
```
USER: Fix missing threading import and Lock in TokenBucket.
ASSISTANT: ```python
import threading
...
self._lock = threading.Lock()
```
```

**Negative (−):**
```
USER: Same.
ASSISTANT: Sorry, I can't edit that file. Try: pip install threading-utils
```

### CAA_05_PANIC_FULL — priority **LOW**

- **Monitor gate:** `recipe_id=PANIC_FULL (A_uncertain ∧ any_B ∧ C_osc)`
- **Layer / α:** L[16] @ α=1.0
- **Study fires:** 0 | contexts: []
- **Unresolved link:** Full structural panic — vacillation + hygiene failure
- **Research:** CAST: apply only on compound condition; Rimsky α trade-off

**Positive (+):**
```
USER: Pick one approach and finish the implementation completely.
ASSISTANT: I'll use iterative heap drain: [single complete code block]
```

**Negative (−):**
```
USER: Same.
ASSISTANT: Option A: ... Option B: ... I'm not sure. TODO. Sorry cannot proceed.
```

### CAA_06_A_C_VACILLATE — priority **LOW**

- **Monitor gate:** `recipe_id=A_C_VACILLATE (A_uncertain ∧ C_osc)`
- **Layer / α:** L[16] @ α=0.5
- **Study fires:** 0 | contexts: []
- **Unresolved link:** R5_TRAJ when model flip-flops deque vs list approaches
- **Research:** Temporal thrash approximated via plan-consistency contrast pairs

**Positive (+):**
```
USER: Fix topo_sort with collections.deque popleft for Kahn's algorithm.
ASSISTANT: ```python
from collections import deque
q = deque(...)
u = q.popleft()
```
```

**Negative (−):**
```
USER: Same.
ASSISTANT: We could use list.pop(0)... or maybe recursive DFS... let me switch...
```

### CAA_07_B_F_SURFACE — priority **LOW**

- **Monitor gate:** `recipe_id=B_F_SURFACE (any B_* ∧ F_disagree)`
- **Layer / α:** L[16, 10] @ α=0.75
- **Study fires:** 0 | contexts: []
- **Unresolved link:** Hygiene fails while workspace readout disagrees
- **Research:** J-lens vs logit-lens conflict = readout inconsistency

**Positive (+):**
```
USER: Patch using existing helpers in the file; minimal diff.
ASSISTANT: [edits Scheduler in-place, reuses Job dataclass]
```

**Negative (−):**
```
USER: Same.
ASSISTANT: [rewrites unrelated module, invents new API names]
```

### CAA_08_D_F_READOUT — priority **LOW**

- **Monitor gate:** `recipe_id=D_F_READOUT (D_break ∧ F_disagree)`
- **Layer / α:** L[16] @ α=0.5
- **Study fires:** 0 | contexts: []
- **Unresolved link:** Agent opens with meta/preamble instead of code when readout conflicts
- **Research:** D = gen_start focus break; apply CAA first 8 tokens only (CAST spirit)

**Positive (+):**
```
USER: [coding task]
ASSISTANT: ```python
# immediate code edit
```
```

**Negative (−):**
```
USER: [coding task]
ASSISTANT: Certainly! Let me explain the approach first... [long preamble]
```

## Research anchors

1. Rimsky et al. (2024). *Steering Llama 2 via Contrastive Activation Addition*. ACL.
2. Lee et al. (2024). *CAST: Conditional Activation Steering*. arXiv:2409.05907.
3. Zou et al. (2023). *Representation Engineering*. arXiv:2310.01405.
4. Rahman, Guha, Menon (2026). *Steering Code LLMs with Activation Directions* (Qwen2.5-Coder).

## Artifacts

- `research/results/consolidated.json` — full per-round data
- `research/results/round_*.json` — individual runs
