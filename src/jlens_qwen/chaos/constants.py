"""Locked constants for the A–G chaos checker (plan v2)."""

PROBE_LAYERS: list[int] = [10, 16, 22]
TOP_K: int = 20
WINDOW_C: int = 16
F2_STRIDE: int = 8
G_DEBOUNCE_TOKENS: int = 32
G_SPARSE_K: int = 25
G_MAX_DICT: int = 2048
F5_MIN_FLAG_CATEGORIES: int = 2

# Per-token budgets (microseconds) — Arc 140V realistic
BUDGET_F0_US: int = 300
BUDGET_F1_P50_US: int = 20_000
BUDGET_F1_P95_US: int = 50_000
BUDGET_F2_US: int = 15_000
BUDGET_F3_US: int = 15_000
BUDGET_F5_F_US: int = 2000
BUDGET_F5_G_US: int = 120_000

# Degradation: if chaos/token exceeds this for N tokens, degrade
DEGRADE_P95_MS: float = 50.0
DEGRADE_STREAK: int = 32

CHAOS_SCHEMA_VERSION: int = 2

# Bump this when persisted chaos JSON shape changes. Restart serve_qwen_openai
# after any bump so finalize() writes the new fields (e.g. token_ids, divergence_hints).
