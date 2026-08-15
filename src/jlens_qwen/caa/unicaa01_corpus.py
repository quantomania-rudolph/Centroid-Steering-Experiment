"""UNI_CAA01 v2 corpus layout — panic/neutral runs + calm-tail slot."""

from __future__ import annotations

from pathlib import Path

# Repo root = parents[3] from src/jlens_qwen/caa/
_REPO = Path(__file__).resolve().parents[3]

CORPUS_ROOT = _REPO / "experiments" / "unicaa01_corpus"
CORPUS_OUT = CORPUS_ROOT / "out"
CORPUS_REGISTRY = CORPUS_ROOT / "registry"
CORPUS_MANIFEST = CORPUS_ROOT / "manifest.json"
CORPUS_SESSIONS = CORPUS_ROOT / "sessions"

CAA_CORPUS_ROOT = _REPO / "artifacts" / "caa" / "UNI_CAA01_corpus"
CALM_TAILS = CAA_CORPUS_ROOT / "calm_tails"
NATURAL_PAIRS = CAA_CORPUS_ROOT / "pairs" / "natural"
NEUTRAL_ARTIFACTS = CAA_CORPUS_ROOT / "neutral"
CURATION_PACKS = CAA_CORPUS_ROOT / "curation_packs"

CORPUS_GLOB = "experiments/unicaa01_corpus/out/*_latest.json"
LEGACY_CORPUS_GLOB = "experiments/out/*_latest.json"

EXPERIMENT_PREFIX = "unicaa01_corpus"


def ensure_dirs() -> None:
    for p in (
        CORPUS_OUT,
        CORPUS_REGISTRY,
        CORPUS_SESSIONS,
        CALM_TAILS,
        NATURAL_PAIRS,
        NEUTRAL_ARTIFACTS,
        CURATION_PACKS,
    ):
        p.mkdir(parents=True, exist_ok=True)
