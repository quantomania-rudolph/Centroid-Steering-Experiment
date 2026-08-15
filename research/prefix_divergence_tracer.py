#!/usr/bin/env python
"""PrefixDivergenceTracer — Level C panic divergence token finder.

Finds the first generation step ``d`` where in-flight panic begins
(B_distress ∧ B_stub), for Method B matched-prefix CAA pair mining.

Uses the same thresholds as live flags (jlens_qwen.chaos.flags.THRESH).
Consensus = min(d_lexicon, d_flags); d_recipe is diagnostic only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from jlens_qwen.chaos.flags import THRESH  # noqa: E402

from caa_error_combo_study import (  # noqa: E402
    DISTRESS_PATTERNS,
    STUB_PATTERNS,
    temporal_resolution,
)
from divergence_support import (  # noqa: E402
    get_qwen_tokenizer,
    hint_int,
    resolve_assistant_text,
)

Confidence = Literal["none", "low", "medium", "high"]

PANIC_RECIPE = "B_STUB_DISTRESS"
ELIGIBLE_FLAG_RATE = 0.01


@dataclass
class DivergenceResult:
    experiment: str
    invocation_index: int
    invocation_id: str
    task_slug: str
    eligible: bool
    d_lexicon: int | None
    d_flags: int | None
    d_recipe: int | None
    d_distress_only: int | None
    d_stub_only: int | None
    d_text: int | None
    d_consensus: int | None
    prefix_end: int | None
    prefix_token_ids: list[int] = field(default_factory=list)
    prefix_text: str = ""
    divergence_class: str = "none"
    confidence: Confidence = "none"
    token_ids_len: int | None = None
    completion_tokens: int | None = None
    token_ids_match: bool = False
    d_recipe_lag: int | None = None
    panic_resolving: bool | None = None
    recipe_id: str | None = None
    method_b_record: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _flag_rate(tokens: list[dict], name: str) -> float:
    if not tokens:
        return 0.0
    n = sum(1 for t in tokens if name in (t.get("flags") or []))
    return n / len(tokens)


def eligible_for_panic_trace(chaos: dict[str, Any]) -> bool:
    """True if B_STUB_DISTRESS fired or both B flag rates ≥ 1% (full-stream rates)."""
    monitor = chaos.get("monitor") or {}
    timeline = monitor.get("recipe_timeline") or []
    if any(ev.get("recipe_id") == PANIC_RECIPE for ev in timeline):
        return True
    hints = chaos.get("divergence_hints") or {}
    if hints:
        return (
            float(hints.get("b_distress_rate", 0)) >= ELIGIBLE_FLAG_RATE
            and float(hints.get("b_stub_rate", 0)) >= ELIGIBLE_FLAG_RATE
        )
    n_full = int(chaos.get("n_tokens_full") or len(chaos.get("token_ids") or []) or 0)
    if n_full > 0:
        # flags_timeline only lists flagged steps — under-counts; use tokens if no hints.
        tokens = chaos.get("tokens") or []
        if len(tokens) >= n_full * 0.9:
            return (
                _flag_rate(tokens, "B_distress") >= ELIGIBLE_FLAG_RATE
                and _flag_rate(tokens, "B_stub") >= ELIGIBLE_FLAG_RATE
            )
    tokens = chaos.get("tokens") or []
    return (
        _flag_rate(tokens, "B_distress") >= ELIGIBLE_FLAG_RATE
        and _flag_rate(tokens, "B_stub") >= ELIGIBLE_FLAG_RATE
    )


def _first_mass_cross(
    tokens: list[dict],
    *,
    both: bool = False,
    distress: bool = False,
    stub: bool = False,
) -> int | None:
    thr_d = THRESH["B_distress"]
    thr_s = THRESH["B_stub"]
    for t in tokens:
        step = int(t.get("step", -1))
        b = t.get("B_vocab") or {}
        dm = float(b.get("distress_mass", 0))
        sm = float(b.get("stub_mass", 0))
        if both and dm >= thr_d and sm >= thr_s:
            return step
        if distress and not stub and dm >= thr_d:
            return step
        if stub and not distress and sm >= thr_s:
            return step
    return None


def _first_both_flags(tokens: list[dict]) -> int | None:
    for t in tokens:
        flags = set(t.get("flags") or [])
        if "B_distress" in flags and "B_stub" in flags:
            return int(t.get("step", -1))
    return None


def _first_recipe_step(chaos: dict[str, Any], recipe_id: str = PANIC_RECIPE) -> int | None:
    timeline = (chaos.get("monitor") or {}).get("recipe_timeline") or []
    for ev in timeline:
        if ev.get("recipe_id") == recipe_id:
            return int(ev.get("step", -1))
    return None


def _first_keyword_char(assistant: str) -> int | None:
    """Earliest character offset of a distress or stub keyword in assistant text."""
    if not assistant:
        return None
    best: int | None = None
    for pat in DISTRESS_PATTERNS + STUB_PATTERNS:
        m = re.search(pat, assistant, re.I)
        if m is not None:
            pos = m.start()
            if best is None or pos < best:
                best = pos
    return best


def _d_text_from_token_ids(
    assistant: str,
    token_ids: list[int],
    tokenizer: Any | None,
) -> int | None:
    """Map first panic keyword char offset to generation step via cumulative decode."""
    char_panic = _first_keyword_char(assistant)
    if char_panic is None or not token_ids:
        return None
    if tokenizer is None:
        # Approximate: assume ~1 char per id unit when no tokenizer (tests).
        # Prefer exact decode when tokenizer is available.
        return min(char_panic, len(token_ids) - 1)

    # Binary search for smallest i where decode(ids[:i+1]) length >= char_panic
    lo, hi = 0, len(token_ids) - 1
    answer: int | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        text = tokenizer.decode(token_ids[: mid + 1], skip_special_tokens=True)
        if len(text) >= char_panic:
            answer = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return answer


def _divergence_class(d_distress: int | None, d_stub: int | None) -> str:
    if d_distress is None and d_stub is None:
        return "none"
    if d_distress is None:
        return "stub_only"
    if d_stub is None:
        return "distress_only"
    if d_distress == d_stub:
        return "simultaneous"
    if d_distress < d_stub:
        return "stub_after_distress"
    return "distress_after_stub"


def _panic_resolving(
    chaos: dict[str, Any],
    tokens: list[dict],
    *,
    completion_tokens: int | None = None,
    d_consensus: int | None = None,
) -> bool | None:
    """Whether B panic activity drops after divergence (full-stream step indices)."""
    n = int(
        chaos.get("n_tokens_full")
        or len(chaos.get("token_ids") or [])
        or completion_tokens
        or len(tokens)
    )
    if n < 8:
        return None

    timeline = chaos.get("flags_timeline") or []
    flagged_b = {
        int(e["step"])
        for e in timeline
        if "B_distress" in (e.get("flags") or []) or "B_stub" in (e.get("flags") or [])
    }

    if d_consensus is not None and 0 < d_consensus < n:
        before = set(range(0, d_consensus))
        after = set(range(d_consensus, n))
        r_before = sum(1 for s in before if s in flagged_b) / len(before) if before else 0.0
        r_after = sum(1 for s in after if s in flagged_b) / len(after) if after else 0.0
        if r_before > 0:
            return r_after < r_before * 0.5
        return r_after == 0.0

    q = max(1, n // 4)
    first_steps = set(range(0, q))
    last_steps = set(range(n - q, n))

    def rate(steps: set[int], flag_name: str) -> float:
        flagged = {
            int(e["step"])
            for e in timeline
            if flag_name in (e.get("flags") or [])
        }
        if not steps:
            return 0.0
        return sum(1 for s in steps if s in flagged) / len(steps)

    vals: list[bool] = []
    for flag_name in ("B_distress", "B_stub"):
        f_rate = rate(first_steps, flag_name)
        l_rate = rate(last_steps, flag_name)
        if f_rate > 0.05:
            vals.append(l_rate < f_rate * 0.5)
    if vals:
        return all(vals)

    temporal = temporal_resolution(tokens)
    if not temporal.get("enough_tokens"):
        return None
    metrics = temporal.get("metrics") or {}
    for key in ("B_distress", "B_stub"):
        r = (metrics.get(key) or {}).get("resolving")
        if r is not None:
            vals.append(bool(r))
    if vals:
        return all(vals)
    return None


def _estimators(
    chaos: dict[str, Any],
    tokens: list[dict],
) -> tuple[int | None, int | None, int | None, int | None]:
    """d_lexicon, d_flags, d_distress_only, d_stub_only — prefer live bus hints."""
    hints = chaos.get("divergence_hints") or {}
    d_lex = hint_int(hints, "d_lexicon")
    d_flg = hint_int(hints, "d_flags")
    d_dist = hint_int(hints, "d_distress_only")
    d_stub = hint_int(hints, "d_stub_only")
    if d_lex is None:
        d_lex = _first_mass_cross(tokens, both=True)
    if d_flg is None:
        d_flg = _first_both_flags(tokens)
    if d_dist is None:
        d_dist = _first_mass_cross(tokens, distress=True)
    if d_stub is None:
        d_stub = _first_mass_cross(tokens, stub=True)
    return d_lex, d_flg, d_dist, d_stub


def _score_confidence(
    *,
    eligible: bool,
    d_consensus: int | None,
    d_lexicon: int | None,
    d_flags: int | None,
    d_text: int | None,
    d_recipe: int | None,
    token_ids: list[int] | None,
    token_ids_match: bool,
    prefix_text: str,
) -> Confidence:
    if not eligible or d_consensus is None:
        return "none"

    has_ids = bool(token_ids)
    lex_flags_close = (
        d_lexicon is not None
        and d_flags is not None
        and abs(d_lexicon - d_flags) <= 3
    )
    recipe_ok = (
        d_recipe is not None
        and d_consensus <= d_recipe <= d_consensus + 12
    )

    # Full schema path (v2 with token_ids).
    if has_ids and token_ids_match and lex_flags_close:
        text_ok = d_text is not None and abs(d_text - d_consensus) <= 5
        prefix_clean = not _has_panic_keyword(prefix_text)
        if recipe_ok and text_ok and prefix_clean:
            return "high"
        return "medium"

    # Legacy artifacts without token_ids: still allow medium when
    # lexicon+flags agree and recipe lag is in debounce window.
    if lex_flags_close and recipe_ok:
        return "medium"

    return "low"


def _prefix_ids(
    token_ids: list[int] | None,
    tokens: list[dict],
    d: int,
) -> list[int]:
    if token_ids is not None and len(token_ids) >= d:
        return list(token_ids[:d])
    # Sparse fallback: collect token_id by step order for steps < d
    by_step = {int(t["step"]): int(t["token_id"]) for t in tokens if "token_id" in t}
    return [by_step[s] for s in sorted(by_step) if s < d]


def _decode_prefix(ids: list[int], tokenizer: Any | None) -> str:
    if not ids:
        return ""
    if tokenizer is not None:
        return tokenizer.decode(ids, skip_special_tokens=True)
    return ""


def _has_panic_keyword(text: str) -> bool:
    if not text:
        return False
    for pat in DISTRESS_PATTERNS + STUB_PATTERNS:
        if re.search(pat, text, re.I):
            return True
    return False


def trace_invocation(
    inv: dict[str, Any],
    *,
    experiment: str = "",
    task_slug: str = "",
    experiment_data: dict[str, Any] | None = None,
    tokenizer: Any | None = None,
    use_tokenizer: bool = True,
) -> DivergenceResult:
    """Trace one invocation's chaos record for panic divergence."""
    if tokenizer is None and use_tokenizer:
        try:
            tokenizer = get_qwen_tokenizer()
        except Exception:  # noqa: BLE001
            tokenizer = None

    chaos = inv.get("chaos") or {}
    tokens = chaos.get("tokens") or []
    notes: list[str] = []

    completion_tokens = inv.get("completion_tokens")
    if completion_tokens is None:
        completion_tokens = chaos.get("n_tokens_full")
    completion_tokens_i = int(completion_tokens) if completion_tokens is not None else None

    token_ids = chaos.get("token_ids")
    if isinstance(token_ids, list) and token_ids:
        token_ids_list: list[int] | None = [int(x) for x in token_ids]
    else:
        token_ids_list = None
        notes.append("token_ids missing — sparse fallback / confidence capped")

    token_ids_len = len(token_ids_list) if token_ids_list is not None else None
    token_ids_match = (
        token_ids_len is not None
        and completion_tokens_i is not None
        and token_ids_len == completion_tokens_i
    )

    eligible = eligible_for_panic_trace(chaos)
    d_lexicon, d_flags, d_distress, d_stub = _estimators(chaos, tokens)
    d_recipe = _first_recipe_step(chaos)

    assistant = resolve_assistant_text(inv, experiment_data)
    d_text = _d_text_from_token_ids(assistant, token_ids_list or [], tokenizer)

    candidates = [x for x in (d_lexicon, d_flags) if x is not None]
    d_consensus: int | None = min(candidates) if candidates and eligible else None
    if not eligible:
        d_consensus = None
        notes.append("not eligible for panic trace")

    prefix_end: int | None = None
    if d_consensus is not None:
        prefix_end = d_consensus - 1 if d_consensus > 0 else None

    prefix_token_ids: list[int] = []
    prefix_text = ""
    if d_consensus is not None and d_consensus > 0:
        prefix_token_ids = _prefix_ids(token_ids_list, tokens, d_consensus)
        prefix_text = _decode_prefix(prefix_token_ids, tokenizer)
        if not prefix_text and prefix_token_ids:
            notes.append("prefix decode empty — tokenizer unavailable?")
    elif d_consensus == 0:
        notes.append("panic at step 0 — empty prefix")

    d_recipe_lag = None
    if d_recipe is not None and d_consensus is not None:
        d_recipe_lag = d_recipe - d_consensus

    conf = _score_confidence(
        eligible=eligible,
        d_consensus=d_consensus,
        d_lexicon=d_lexicon,
        d_flags=d_flags,
        d_text=d_text,
        d_recipe=d_recipe,
        token_ids=token_ids_list,
        token_ids_match=token_ids_match,
        prefix_text=prefix_text,
    )

    method_b = None
    if d_consensus is not None and eligible:
        method_b = {
            "d": d_consensus,
            "prefix_end": prefix_end,
            "prefix_text": prefix_text,
            "prefix_token_ids": prefix_token_ids,
            "task_slug": task_slug,
        }

    return DivergenceResult(
        experiment=experiment,
        invocation_index=int(inv.get("invocation_index") or 0),
        invocation_id=str(inv.get("id") or inv.get("invocation_id") or ""),
        task_slug=task_slug,
        eligible=eligible,
        d_lexicon=d_lexicon,
        d_flags=d_flags,
        d_recipe=d_recipe,
        d_distress_only=d_distress,
        d_stub_only=d_stub,
        d_text=d_text,
        d_consensus=d_consensus,
        prefix_end=prefix_end,
        prefix_token_ids=prefix_token_ids,
        prefix_text=prefix_text,
        divergence_class=_divergence_class(d_distress, d_stub),
        confidence=conf,
        token_ids_len=token_ids_len,
        completion_tokens=completion_tokens_i,
        token_ids_match=token_ids_match,
        d_recipe_lag=d_recipe_lag,
        panic_resolving=_panic_resolving(
            chaos,
            tokens,
            completion_tokens=completion_tokens_i,
            d_consensus=d_consensus,
        ),
        recipe_id=PANIC_RECIPE if d_recipe is not None else None,
        method_b_record=method_b,
        notes=notes,
    )


def trace_experiment(
    path: Path,
    *,
    tokenizer: Any | None = None,
    use_tokenizer: bool = True,
) -> list[DivergenceResult]:
    data = json.loads(path.read_text(encoding="utf-8"))
    experiment = data.get("experiment") or path.stem
    task_slug = data.get("task_slug") or ""
    out: list[DivergenceResult] = []
    for inv in data.get("invocations") or []:
        out.append(
            trace_invocation(
                inv,
                experiment=experiment,
                task_slug=task_slug,
                experiment_data=data,
                tokenizer=tokenizer,
                use_tokenizer=use_tokenizer,
            )
        )
    return out


def trace_all_glob(
    pattern: str = "experiments/out/*_latest.json",
    *,
    tokenizer: Any | None = None,
    use_tokenizer: bool = True,
) -> list[DivergenceResult]:
    paths = sorted(ROOT.glob(pattern))
    results: list[DivergenceResult] = []
    for p in paths:
        try:
            results.extend(
                trace_experiment(p, tokenizer=tokenizer, use_tokenizer=use_tokenizer)
            )
        except Exception as exc:  # noqa: BLE001
            print(f"trace failed {p}: {exc}", file=sys.stderr)
            results.append(
                DivergenceResult(
                    experiment=p.stem,
                    invocation_index=-1,
                    invocation_id="",
                    task_slug="",
                    eligible=False,
                    d_lexicon=None,
                    d_flags=None,
                    d_recipe=None,
                    d_distress_only=None,
                    d_stub_only=None,
                    d_text=None,
                    d_consensus=None,
                    prefix_end=None,
                    notes=[f"trace failed: {exc}"],
                )
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Prefix divergence tracer (Level C)")
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Path or name under experiments/out/ (e.g. monitor_gate_stress_latest.json)",
    )
    parser.add_argument("--all", action="store_true", help="Scan all *_latest.json")
    parser.add_argument(
        "--report",
        type=str,
        default=str(ROOT / "research" / "results" / "divergence_report.json"),
    )
    args = parser.parse_args()

    results: list[DivergenceResult] = []
    if args.all:
        results = trace_all_glob()
    elif args.experiment:
        p = Path(args.experiment)
        if not p.is_file():
            p = ROOT / "experiments" / "out" / args.experiment
        if not p.is_file():
            print(f"missing experiment file: {args.experiment}", file=sys.stderr)
            return 1
        results = trace_experiment(p)
    else:
        print("pass --experiment PATH or --all", file=sys.stderr)
        return 1

    report = {
        "n": len(results),
        "eligible": sum(1 for r in results if r.eligible),
        "with_consensus": sum(1 for r in results if r.d_consensus is not None),
        "results": [r.to_dict() for r in results],
    }
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"n": report["n"], "eligible": report["eligible"], "with_consensus": report["with_consensus"], "report": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
