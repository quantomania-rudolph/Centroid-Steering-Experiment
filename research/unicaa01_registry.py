"""UNICAA01 corpus pair registry — wave 1 (p01–p16) + wave 2 (p17–p66)."""

from __future__ import annotations

from unicaa01_batch2_tasks import PAIRS_BATCH2
from unicaa01_pair_defs import UniCaaPair, neutral_capture_user_prompt
from unicaa01_tasks import PAIRS as PAIRS_WAVE1
from unicaa01_tasks import neutral_capture_specs as _neutral_capture_specs_wave1


PAIRS_ALL: list[UniCaaPair] = list(PAIRS_WAVE1) + list(PAIRS_BATCH2)


def pairs_for_wave(wave: int | None) -> list[UniCaaPair]:
    if wave == 1:
        return list(PAIRS_WAVE1)
    if wave == 2:
        return list(PAIRS_BATCH2)
    return list(PAIRS_ALL)


def get_pair(pair_id: str) -> UniCaaPair:
    for p in PAIRS_ALL:
        if p.pair_id == pair_id:
            return p
    raise KeyError(pair_id)


def neutral_capture_specs(pair_ids: list[str] | None = None) -> list[dict[str, str]]:
    ids = pair_ids or [p.pair_id for p in PAIRS_ALL]
    out: list[dict[str, str]] = []
    for pid in ids:
        p = get_pair(pid)
        out.append(
            {
                "slug": p.neutral_slug,
                "provenance": p.provenance,
                "user_prompt": neutral_capture_user_prompt(p.neutral_message),
                "assistant_start": p.assistant_start,
                "pair_id": p.pair_id,
                "domain": p.domain,
            }
        )
    return out
