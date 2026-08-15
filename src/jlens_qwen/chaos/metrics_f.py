"""F — J-lens vs logit-lens disagreement."""

from __future__ import annotations

from typing import Any


def topm_overlap(ids_j: list[int], ids_ll: list[int], m: int = 5) -> float:
    """|S_J ∩ S_LL| / m using first m of each list."""
    sj = set(ids_j[:m])
    sll = set(ids_ll[:m])
    if m <= 0:
        return 0.0
    return len(sj & sll) / float(m)


def kendall_tau_shared(ids_j: list[int], ids_ll: list[int], m: int = 5) -> float:
    """Kendall τ on shared tokens among top-m; 0 if fewer than 2 shared."""
    sj = ids_j[:m]
    sll = ids_ll[:m]
    shared = [t for t in sj if t in sll]
    if len(shared) < 2:
        return 0.0
    rank_j = {t: i for i, t in enumerate(sj)}
    rank_ll = {t: i for i, t in enumerate(sll)}
    concord = 0
    discord = 0
    for i in range(len(shared)):
        for j in range(i + 1, len(shared)):
            a, b = shared[i], shared[j]
            same = (rank_j[a] - rank_j[b]) * (rank_ll[a] - rank_ll[b]) > 0
            if same:
                concord += 1
            else:
                discord += 1
    total = concord + discord
    if total == 0:
        return 0.0
    return (concord - discord) / total


def lens_disagreement(
    ids_j: list[int],
    ids_ref: list[int],
    *,
    m: int = 5,
) -> dict[str, float]:
    """Top-m overlap between J-lens readout and a reference distribution.

    Production reference is the **sampling-head** top-k (actual decode logits).
    Naive same-layer logit-lens is optional diagnostic only — with a fitted J at
  mid-layer it is expected to diverge even on healthy runs.
    """
    return {
        "overlap": topm_overlap(ids_j, ids_ref, m=m),
        "kendall_tau": kendall_tau_shared(ids_j, ids_ref, m=m),
        "m": float(m),
    }


def f_alarm_from_metrics(f: dict[str, float] | None) -> bool:
    """True when J-lens vs sampling-head overlap is below calibrated threshold."""
    from jlens_qwen.chaos.flags import THRESH

    if not isinstance(f, dict):
        return False
    ov = f.get("overlap_sample")
    if ov is None:
        # Legacy runs stored naive logit-lens overlap only — known false positive.
        return False
    return float(ov) <= THRESH["F_low_overlap"]
