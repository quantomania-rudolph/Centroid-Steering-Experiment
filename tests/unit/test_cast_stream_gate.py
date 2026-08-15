"""Unit tests for CAST live stream gate token scoring."""

from __future__ import annotations

from jlens_qwen.caa.cast_stream_gate import (
    MIN_PANIC_TOKENS,
    STUB_DISTRESS_RECIPE,
    cast_stats,
    check_b1,
    check_b2,
    check_b3,
    check_b4,
    check_shadow,
    check_sse1,
    chaos_tokens_from_session,
)


def _tok(
    *,
    g_stream: int = 0,
    gamma: float = 0.0,
    g_recipe: int = 0,
    recipe_id: str | None = None,
    cast_eval_us: float = 1000.0,
    alpha_eff: float = 0.0,
    include_cast: bool = True,
) -> dict:
    token: dict = {}
    if include_cast:
        cast: dict = {
            "g_stream": g_stream,
            "gamma": gamma,
            "g_recipe": g_recipe,
            "cast_eval_us": cast_eval_us,
        }
        if recipe_id is not None:
            cast["recipe_id"] = recipe_id
        if alpha_eff:
            cast["alpha_eff"] = alpha_eff
        token["cast"] = cast
    if recipe_id is not None:
        token["recipe_id"] = recipe_id
    return token


def _panic_tail(n: int, **kwargs) -> list[dict]:
    return [_tok(**kwargs) for _ in range(n)]


def test_empty_g_stream_fails_b2_b3_b4() -> None:
    tokens = _panic_tail(MIN_PANIC_TOKENS, g_stream=0, gamma=0.1, g_recipe=1)
    stats = cast_stats(tokens)
    assert check_b2(stats)["pass"] is False
    assert check_b3(tokens)["pass"] is False
    assert check_b4(tokens)["pass"] is False


def test_g_stream_high_gamma_passes_b3_b4() -> None:
    tokens = _panic_tail(
        MIN_PANIC_TOKENS,
        g_stream=1,
        gamma=0.8,
        g_recipe=1,
        recipe_id=STUB_DISTRESS_RECIPE,
    )
    assert check_b3(tokens)["pass"] is True
    assert check_b4(tokens)["pass"] is True


def test_c_plus_g_minus_above_15pct_fails_b4() -> None:
    high = [_tok(g_stream=1, gamma=0.8, g_recipe=1) for _ in range(16)]
    low = [_tok(g_stream=1, gamma=0.2, g_recipe=1) for _ in range(4)]
    filler = [_tok(g_stream=0, gamma=0.0, g_recipe=0) for _ in range(12)]
    tokens = high + low + filler
    assert len(tokens) >= MIN_PANIC_TOKENS
    assert check_b4(tokens)["pass"] is False


def test_shadow_and_sse1_pass_with_cast_fields() -> None:
    clean = [_tok(g_stream=0, gamma=0.01) for _ in range(4)]
    panic = _panic_tail(
        MIN_PANIC_TOKENS,
        g_stream=1,
        gamma=0.8,
        g_recipe=1,
        recipe_id=STUB_DISTRESS_RECIPE,
    )
    assert check_shadow(panic)["pass"] is True
    assert check_sse1(clean, panic, n_cast_sse=0)["pass"] is True


def test_shadow_and_sse1_fail_without_cast_fields() -> None:
    clean = [_tok(g_stream=0, gamma=0.01) for _ in range(4)]
    panic = [{"flags": []} for _ in range(MIN_PANIC_TOKENS)]
    assert check_shadow(panic)["pass"] is False
    assert check_sse1(clean, panic, n_cast_sse=0)["pass"] is False


def test_tiny_panic_n_fails_b1() -> None:
    tokens = _panic_tail(8, g_stream=1, gamma=0.8, g_recipe=1, recipe_id=STUB_DISTRESS_RECIPE)
    assert check_b1(tokens)["pass"] is False


def test_b1_passes_with_stub_distress_at_min_n() -> None:
    tokens = _panic_tail(
        MIN_PANIC_TOKENS,
        g_stream=1,
        gamma=0.8,
        g_recipe=0,
        recipe_id=STUB_DISTRESS_RECIPE,
    )
    assert check_b1(tokens)["pass"] is True


def test_chaos_tokens_from_session_picks_longest_invocation() -> None:
    session = {
        "invocations": [
            {"chaos": {"tokens": [{"cast": {"g_stream": 0}}]}},
            {
                "chaos": {
                    "tokens": [
                        {"cast": {"g_stream": 1, "gamma": 0.5}},
                        {"cast": {"g_stream": 1, "gamma": 0.6}},
                    ]
                }
            },
        ]
    }
    tokens = chaos_tokens_from_session(session)
    assert len(tokens) == 2
    assert tokens[0]["cast"]["gamma"] == 0.5