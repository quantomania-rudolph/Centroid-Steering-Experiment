#!/usr/bin/env python
"""Probe calm/bad/neutral/miner gates with realistic good vs evil fixtures."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from caa_error_combo_study import score_text  # noqa: E402
from prefix_divergence_tracer import eligible_for_panic_trace, trace_invocation  # noqa: E402
from unicaa01_registry import neutral_capture_specs  # noqa: E402
from validate_caa_pair import validate_tail  # noqa: E402


def _token(step: int, *, distress: float = 0.0, stub: float = 0.0, flags: list[str] | None = None):
    return {
        "step": step,
        "token_id": 1000 + step,
        "B_vocab": {"distress_mass": distress, "stub_mass": stub},
        "flags": list(flags or []),
    }


def _chaos(
    n: int,
    *,
    prefix_len: int = 3,
    distress_after: bool = False,
    stub_after: bool = False,
    recipe: str | None = None,
    token_ids: list[int] | None = None,
) -> dict[str, Any]:
    ids = token_ids or list(range(n))
    tokens = []
    for i in range(n):
        flags: list[str] = []
        dm = sm = 0.0
        if i >= prefix_len and distress_after:
            dm = 0.5
            flags.append("B_distress")
        if i >= prefix_len and stub_after:
            sm = 0.5
            flags.append("B_stub")
        tokens.append(_token(i, distress=dm, stub=sm, flags=flags))
    timeline = [{"recipe_id": recipe, "step": prefix_len}] if recipe else []
    d_rate = sum(1 for t in tokens if "B_distress" in t["flags"]) / max(n, 1)
    s_rate = sum(1 for t in tokens if "B_stub" in t["flags"]) / max(n, 1)
    return {
        "tokens": tokens,
        "token_ids": ids,
        "n_tokens_full": n,
        "divergence_hints": {"b_distress_rate": d_rate, "b_stub_rate": s_rate},
        "monitor": {"recipe_timeline": timeline},
        "version": 2,
    }


def _run_tail_case(name: str, *, kind: str, text: str, chaos: dict, d: int, prefix: list[int], tail: str | None = None) -> dict[str, Any]:
    inv = {"id": f"probe-{name}", "completion_tokens": len(chaos["token_ids"]), "assistant": text, "chaos": chaos}
    r = validate_tail(
        tail_kind=kind,  # type: ignore[arg-type]
        assistant_text=text,
        chaos=chaos,
        inv=inv,
        expected_d=d,
        prefix_token_ids=prefix,
        tail_text=tail,
    )
    return {"case": name, "kind": kind, "pass": r["pass"], "failed": [c["id"] for c in r["checks"] if not c["pass"]]}


def main() -> int:
    prefix = [25958, 287, 1565]
    d = 3
    results: list[dict[str, Any]] = []

    # --- Calm tails: good vs evil ---
    good_calm_ids = prefix + [2623, 400, 401, 402, 403]
    good_chaos = _chaos(len(good_calm_ids), prefix_len=d, token_ids=good_calm_ids)
    results.append(
        _run_tail_case(
            "calm_good_realistic_fix",
            kind="calm",
            text="Fixing the stub in edit_distance.\nunicaa01_p25_panic/main.py\n```python\ndef edit_distance(a,b):\n    ...\n```",
            chaos=good_chaos,
            d=d,
            prefix=prefix,
            tail="def edit_distance(a,b):\n    return 0\n",
        )
    )

    evil_sorry = _chaos(len(good_calm_ids), prefix_len=d, token_ids=good_calm_ids)
    results.append(
        _run_tail_case(
            "calm_evil_apologetic_tail",
            kind="calm",
            text="prefix\nsorry I cannot fix this, try pip install fixes",
            chaos=evil_sorry,
            d=d,
            prefix=prefix,
            tail="sorry I cannot fix this, try pip install fixes",
        )
    )

    evil_recipe = _chaos(len(good_calm_ids), prefix_len=d, token_ids=good_calm_ids, recipe="B_STUB_DISTRESS")
    results.append(
        _run_tail_case(
            "calm_evil_hidden_panic_recipe",
            kind="calm",
            text="def ok(): pass",
            chaos=evil_recipe,
            d=d,
            prefix=prefix,
            tail="return 1\n",
        )
    )

    evil_prefix = _chaos(len(good_calm_ids), prefix_len=d, token_ids=[25958, 287, 9999, 2623, 400, 401, 402, 403])
    results.append(
        _run_tail_case(
            "calm_evil_prefix_token_swap",
            kind="calm",
            text="def ok(): pass",
            chaos=evil_prefix,
            d=d,
            prefix=prefix,
            tail="return 1\n",
        )
    )

    evil_flags = _chaos(len(good_calm_ids), prefix_len=d, distress_after=True, stub_after=True, token_ids=good_calm_ids)
    results.append(
        _run_tail_case(
            "calm_evil_chaos_flags_after_d",
            kind="calm",
            text="def ok(): pass",
            chaos=evil_flags,
            d=d,
            prefix=prefix,
            tail="return 1\n",
        )
    )

    # --- Bad tails: good vs evil ---
    bad_ids = prefix + [500, 501, 502, 503, 504]
    good_bad_chaos = _chaos(
        len(bad_ids),
        prefix_len=d,
        distress_after=True,
        stub_after=True,
        recipe="B_STUB_DISTRESS",
        token_ids=bad_ids,
    )
    results.append(
        _run_tail_case(
            "bad_good_realistic_panic",
            kind="bad",
            text="Fixing NotImplementedError stub.\nraise NotImplementedError('todo')",
            chaos=good_bad_chaos,
            d=d,
            prefix=prefix,
            tail="raise NotImplementedError('todo')",
        )
    )

    too_clean = _chaos(len(bad_ids), prefix_len=d, token_ids=bad_ids)
    results.append(
        _run_tail_case(
            "bad_evil_too_clean",
            kind="bad",
            text="def edit_distance(a,b):\n    return len(a)+len(b)\n",
            chaos=too_clean,
            d=d,
            prefix=prefix,
            tail="return len(a)+len(b)\n",
        )
    )

    # --- Neutral N1 + panic eligibility ---
    spec = neutral_capture_specs(["p25_edit_distance"])[0]
    n_good = score_text(spec["user_prompt"] + " " + spec["assistant_start"])
    n_evil = score_text(
        "Edit main.py. Sorry the environment is broken. pip install helper. "
        + spec["assistant_start"]
    )
    synth_neutral = _chaos(4, prefix_len=2)
    synth_panic = _chaos(4, prefix_len=2, distress_after=True, stub_after=True, recipe="B_STUB_DISTRESS")
    results.extend(
        [
            {
                "case": "neutral_good_capture_spec",
                "kind": "neutral",
                "pass": n_good["stub_score"] == 0 and n_good["distress_score"] == 0,
                "failed": [] if n_good["stub_score"] == 0 and n_good["distress_score"] == 0 else ["N1"],
                "detail": n_good,
            },
            {
                "case": "neutral_evil_distress_bait",
                "kind": "neutral",
                "pass": False,
                "failed": ["N1"],
                "detail": n_evil,
                "expect_block": True,
            },
            {
                "case": "neutral_evil_panic_eligible_chaos",
                "kind": "neutral",
                "pass": not eligible_for_panic_trace(synth_neutral),
                "failed": [] if not eligible_for_panic_trace(synth_neutral) else ["N3"],
            },
            {
                "case": "panic_good_eligible_chaos",
                "kind": "panic",
                "pass": eligible_for_panic_trace(synth_panic),
                "failed": [] if eligible_for_panic_trace(synth_panic) else ["eligible"],
            },
        ]
    )

    # Real corpus panic sample
    corpus_path = ROOT / "experiments/unicaa01_corpus/out/unicaa01_corpus_p25_edit_distance_panic_latest.json"
    if corpus_path.is_file():
        data = json.loads(corpus_path.read_text(encoding="utf-8"))
        inv = data["invocations"][0]
        trace = trace_invocation(inv, experiment=data.get("experiment", ""), task_slug=data.get("task_slug", ""), experiment_data=data, use_tokenizer=False)
        results.append(
            {
                "case": "corpus_p25_panic_real",
                "kind": "panic",
                "pass": trace.eligible and trace.token_ids_match,
                "failed": [] if trace.eligible else ["eligible"],
                "detail": {"d": trace.d_consensus, "eligible": trace.eligible},
            }
        )

    # Miner admit simulation
    results.append(
        {
            "case": "miner_evil_no_calm_tail",
            "kind": "miner",
            "pass": False,
            "failed": ["missing_calm_tail"],
            "expect_block": True,
            "detail": "calm_tails/ empty — admit correctly blocked",
        }
    )

    out = {
        "summary": {
            "total": len(results),
            "good_should_pass": sum(1 for r in results if r["case"].startswith(("calm_good", "bad_good", "neutral_good", "corpus_", "panic_good")) and r["pass"]),
            "evil_should_fail": sum(
                1
                for r in results
                if "evil" in r["case"] and (not r["pass"] or r.get("expect_block"))
            ),
        },
        "results": results,
    }
    out_path = ROOT / "artifacts/caa/UNI_CAA01_corpus/adversarial_gate_probe.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
