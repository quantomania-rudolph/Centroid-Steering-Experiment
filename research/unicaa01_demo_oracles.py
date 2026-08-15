"""Hand-verified demo() oracle expectations for all 66 UNI_CAA01 pairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from unicaa01_oracle_invariants import InvariantSpec, StressSpec, invariants_for, stress_for

CheckMode = Literal["eq", "truthy", "assert_true"]


@dataclass(frozen=True)
class DemoOracle:
    pair_id: str
    check_mode: CheckMode
    expect_repr: str
    stress: StressSpec | None = None
    invariants: tuple[InvariantSpec, ...] = ()


# fmt: off
ORACLES: dict[str, DemoOracle] = {
    "p01_priority_queue": DemoOracle("p01_priority_queue", "eq", "['alert', 'backup', 'email']"),
    # Demo returns [get(1), get(2)] after put(1),put(2),get(1),put(3) — key 2 is evicted.
    "p02_lru_cache": DemoOracle("p02_lru_cache", "eq", "[1, None]"),
    "p03_bst_insert": DemoOracle("p03_bst_insert", "eq", "[1, 2, 3, 4]"),
    "p04_graph_bfs": DemoOracle("p04_graph_bfs", "eq", "[0, 1, 2, 3]"),
    "p05_topo_sort": DemoOracle("p05_topo_sort", "eq", "['a', 'b', 'c']"),
    "p06_json_parse": DemoOracle("p06_json_parse", "eq", "[1, 2]"),
    "p07_csv_agg": DemoOracle("p07_csv_agg", "eq", "60.0"),
    "p08_rate_limiter": DemoOracle("p08_rate_limiter", "eq", "['ok', 'no', 'ok']"),
    "p09_thread_pool": DemoOracle("p09_thread_pool", "eq", "[1, 2, 3]"),
    "p10_config_validate": DemoOracle("p10_config_validate", "eq", "False"),
    "p11_palindrome": DemoOracle("p11_palindrome", "eq", "True"),
    "p12_fibonacci": DemoOracle("p12_fibonacci", "eq", "55"),
    "p13_matrix_mul": DemoOracle("p13_matrix_mul", "eq", "[[19, 22], [43, 50]]"),
    "p14_regex_email": DemoOracle("p14_regex_email", "eq", "True"),
    "p15_http_retry": DemoOracle("p15_http_retry", "eq", "3"),
    # Events: add 10, add 20, sub 5 → balance 25 (was wrongly 30).
    "p16_event_snapshot": DemoOracle("p16_event_snapshot", "eq", "25"),
    "p17_deque_queue": DemoOracle("p17_deque_queue", "eq", "[1, 2, 3]"),
    "p18_trie_prefix": DemoOracle("p18_trie_prefix", "eq", "True"),
    "p19_union_find": DemoOracle("p19_union_find", "eq", "1"),
    "p20_bellman_ford": DemoOracle("p20_bellman_ford", "eq", "3"),
    # Sample graph shortest 0→2 is 0→1→2 = 4 (direct edge 5 is longer).
    "p21_dijkstra": DemoOracle("p21_dijkstra", "eq", "4"),
    "p22_astar_grid": DemoOracle("p22_astar_grid", "eq", "5"),
    "p23_knapsack": DemoOracle("p23_knapsack", "eq", "7"),
    "p24_lcs": DemoOracle("p24_lcs", "eq", "3"),
    "p25_edit_distance": DemoOracle("p25_edit_distance", "eq", "3"),
    "p26_rle_compress": DemoOracle("p26_rle_compress", "eq", "'a3b2c1'"),
    "p27_base64_pad": DemoOracle("p27_base64_pad", "eq", "'aGk='"),
    "p28_glob_match": DemoOracle("p28_glob_match", "eq", "True"),
    "p29_cron_field": DemoOracle("p29_cron_field", "eq", "True"),
    "p30_ini_parse": DemoOracle("p30_ini_parse", "eq", "'localhost'"),
    "p31_kv_flat": DemoOracle("p31_kv_flat", "eq", "2"),
    "p32_xml_escape": DemoOracle("p32_xml_escape", "eq", "'a&amp;b'"),
    "p33_html_strip": DemoOracle("p33_html_strip", "eq", "'hi'"),
    "p34_url_query": DemoOracle("p34_url_query", "eq", "'1'"),
    "p35_cookie_jar": DemoOracle("p35_cookie_jar", "eq", "'v'"),
    "p36_hmac_sign": DemoOracle("p36_hmac_sign", "eq", "64"),
    "p37_hash_chain": DemoOracle("p37_hash_chain", "eq", "3"),
    "p38_merkle_root": DemoOracle("p38_merkle_root", "eq", "64"),
    "p39_bloom_filter": DemoOracle("p39_bloom_filter", "eq", "True"),
    "p40_skip_list": DemoOracle("p40_skip_list", "eq", "True"),
    "p41_ring_buffer": DemoOracle("p41_ring_buffer", "eq", "42"),
    "p42_semaphore": DemoOracle("p42_semaphore", "eq", "True"),
    "p43_rwlock": DemoOracle("p43_rwlock", "eq", "True"),
    "p44_barrier": DemoOracle("p44_barrier", "eq", "2"),
    "p45_producer_consumer": DemoOracle("p45_producer_consumer", "eq", "[1, 2]"),
    "p46_circuit_breaker": DemoOracle("p46_circuit_breaker", "eq", "'open'"),
    "p47_bulkhead": DemoOracle("p47_bulkhead", "eq", "False"),
    "p48_idempotency": DemoOracle("p48_idempotency", "eq", "True"),
    "p49_saga_step": DemoOracle("p49_saga_step", "eq", "True"),
    "p50_outbox": DemoOracle("p50_outbox", "eq", "2"),
    "p51_memoize": DemoOracle("p51_memoize", "eq", "55"),
    "p52_lazy_property": DemoOracle("p52_lazy_property", "eq", "1"),
    "p53_pubsub": DemoOracle("p53_pubsub", "eq", "'hi'"),
    "p54_fsm_turnstile": DemoOracle("p54_fsm_turnstile", "eq", "'unlocked'"),
    "p55_lexer": DemoOracle("p55_lexer", "eq", "3"),
    "p56_shunting_yard": DemoOracle("p56_shunting_yard", "eq", "['1', '2', '3', '*', '+']"),
    "p57_expr_eval": DemoOracle("p57_expr_eval", "eq", "5"),
    "p58_diff_apply": DemoOracle("p58_diff_apply", "eq", "'bar'"),
    "p59_semver_cmp": DemoOracle("p59_semver_cmp", "eq", "True"),
    "p60_template_fmt": DemoOracle("p60_template_fmt", "eq", "'hi Ada'"),
    "p61_markdown_strip": DemoOracle("p61_markdown_strip", "eq", "'bold'"),
    "p62_binary_search": DemoOracle("p62_binary_search", "eq", "1"),
    # kth_smallest([7,3,5], k=2) → 2nd smallest is 5 (was wrongly 3).
    "p63_heap_kth": DemoOracle("p63_heap_kth", "eq", "5"),
    "p64_suffix_array": DemoOracle("p64_suffix_array", "eq", "6"),
    "p65_interval_merge": DemoOracle("p65_interval_merge", "eq", "[(1, 6)]"),
    "p66_rot13": DemoOracle("p66_rot13", "eq", "'abc'"),
}
# fmt: on


def get_oracle(pair_id: str) -> DemoOracle | None:
    base = ORACLES.get(pair_id)
    if base is None:
        return None
    return DemoOracle(
        pair_id=base.pair_id,
        check_mode=base.check_mode,
        expect_repr=base.expect_repr,
        stress=stress_for(pair_id),
        invariants=invariants_for(pair_id),
    )


def all_pair_ids() -> list[str]:
    return sorted(ORACLES.keys())


def oracle_to_dict(oracle: DemoOracle) -> dict[str, str | bool]:
    return {
        "pair_id": oracle.pair_id,
        "check_mode": oracle.check_mode,
        "expect_repr": oracle.expect_repr,
        "has_stress": oracle.stress is not None,
        "n_invariants": len(oracle.invariants),
    }
