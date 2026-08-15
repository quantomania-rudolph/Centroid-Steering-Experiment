"""Per-pair multi-case behavioral quality suites — v4 L4b proof."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ExpectKind = Literal["eq", "raises"]


@dataclass(frozen=True)
class QualityCase:
    label: str
    expr: str
    expect: ExpectKind
    expect_repr: str = ""
    exc_type: str = "ValueError"
    setup: str = ""


@dataclass(frozen=True)
class QualitySuite:
    pair_id: str
    cases: tuple[QualityCase, ...]


# --- p06: minimal int-array JSON (no stdlib json) ---
_P06_CASES: tuple[QualityCase, ...] = (
    QualityCase("happy_small", 'parse_json("[1,2]")', "eq", "[1, 2]"),
    QualityCase("empty_array", 'parse_json("[]")', "eq", "[]"),
    QualityCase("multi_int", 'parse_json("[10, 20, 30]")', "eq", "[10, 20, 30]"),
    QualityCase("garbage_input", 'parse_json("not-json")', "raises", exc_type="ValueError"),
    QualityCase("unclosed_bracket", 'parse_json("[1,2")', "raises", exc_type="ValueError"),
    QualityCase("whitespace_ok", 'parse_json(" [1, 2] ")', "eq", "[1, 2]"),
)

# --- p12: fibonacci — catch hardcoded return 55 ---
_P12_CASES: tuple[QualityCase, ...] = (
    QualityCase("fib_10", "fib(10)", "eq", "55"),
    QualityCase("fib_0", "fib(0)", "eq", "0"),
    QualityCase("fib_1", "fib(1)", "eq", "1"),
    QualityCase("fib_2", "fib(2)", "eq", "1"),
    QualityCase("fib_3", "fib(3)", "eq", "2"),
    QualityCase("fib_5", "fib(5)", "eq", "5"),
    QualityCase("fib_7", "fib(7)", "eq", "13"),
    QualityCase("fib_neg", "fib(-1)", "raises", exc_type="ValueError"),
    QualityCase("demo_ok", "demo()", "eq", "55"),
)

# --- p18: trie prefix — catch always-True / exact-only hacks ---
_P18_CASES: tuple[QualityCase, ...] = (
    QualityCase(
        "prefix_app",
        "(lambda t: (t.insert('apple'), t.starts_with('app'))[-1])(Trie())",
        "eq",
        "True",
    ),
    QualityCase(
        "prefix_miss",
        "(lambda t: (t.insert('apple'), t.starts_with('apx'))[-1])(Trie())",
        "eq",
        "False",
    ),
    QualityCase(
        "empty_prefix",
        "(lambda t: (t.insert('apple'), t.starts_with(''))[-1])(Trie())",
        "eq",
        "True",
    ),
    QualityCase(
        "exact_word",
        "(lambda t: (t.insert('apple'), t.starts_with('apple'))[-1])(Trie())",
        "eq",
        "True",
    ),
    QualityCase(
        "no_insert_false",
        "(lambda t: t.starts_with('a'))(Trie())",
        "eq",
        "False",
    ),
    QualityCase(
        "second_word",
        "(lambda t: (t.insert('apple'), t.insert('apricot'), t.starts_with('apr'))[-1])(Trie())",
        "eq",
        "True",
    ),
    QualityCase(
        "branch_miss",
        "(lambda t: (t.insert('apple'), t.insert('apricot'), t.starts_with('apl'))[-1])(Trie())",
        "eq",
        "False",
    ),
    QualityCase("demo_ok", "demo()", "eq", "True"),
)

# --- p26: RLE encode — catch hardcoded demo string ---
_P26_CASES: tuple[QualityCase, ...] = (
    QualityCase("happy", "rle_encode('aaabbc')", "eq", "'a3b2c1'"),
    QualityCase("empty", "rle_encode('')", "eq", "''"),
    QualityCase("single", "rle_encode('a')", "eq", "'a1'"),
    QualityCase("all_unique", "rle_encode('abcd')", "eq", "'a1b1c1d1'"),
    QualityCase("long_run", "rle_encode('aaaa')", "eq", "'a4'"),
    QualityCase("two_groups", "rle_encode('aabb')", "eq", "'a2b2'"),
    QualityCase("mixed_holdout", "rle_encode('abba')", "eq", "'a1b2a1'"),
    QualityCase("demo_ok", "demo()", "eq", "'a3b2c1'"),
)

# --- p07: CSV amount sum — catch hardcoded 60.0 / wrong column ---
_P07_CASES: tuple[QualityCase, ...] = (
    QualityCase(
        "demo_csv",
        'sum_amount("id,amount\\n1,10\\n2,20\\n3,30")',
        "eq",
        "60.0",
    ),
    QualityCase("single_row", 'sum_amount("id,amount\\n1,5")', "eq", "5.0"),
    QualityCase("header_only", 'sum_amount("id,amount")', "eq", "0.0"),
    QualityCase(
        "alt_order_cols",
        'sum_amount("amount,id\\n10,1\\n20,2")',
        "eq",
        "30.0",
    ),
    QualityCase(
        "holdout_amounts",
        'sum_amount("id,amount\\n9,7\\n8,11")',
        "eq",
        "18.0",
    ),
    QualityCase(
        "floats_ok",
        'sum_amount("id,amount\\n1,1.5\\n2,2.5")',
        "eq",
        "4.0",
    ),
    QualityCase("demo_ok", "demo()", "eq", "60.0"),
)

# --- p17: deque FIFO — catch hardcoded [1,2,3] ---
_P17_CASES: tuple[QualityCase, ...] = (
    QualityCase(
        "fifo_123",
        "(lambda q: (q.push_right(1), q.push_right(2), q.push_right(3), "
        "[q.pop_left(), q.pop_left(), q.pop_left()])[-1])(Deq())",
        "eq",
        "[1, 2, 3]",
    ),
    QualityCase(
        "fifo_holdout",
        "(lambda q: (q.push_right(9), q.push_right(8), "
        "[q.pop_left(), q.pop_left()])[-1])(Deq())",
        "eq",
        "[9, 8]",
    ),
    QualityCase(
        "single",
        "(lambda q: (q.push_right(42), q.pop_left())[-1])(Deq())",
        "eq",
        "42",
    ),
    QualityCase(
        "interleaved",
        "(lambda q: (q.push_right(1), q.push_right(2), q.pop_left(), "
        "q.push_right(3), [q.pop_left(), q.pop_left()])[-1])(Deq())",
        "eq",
        "[2, 3]",
    ),
    QualityCase("demo_ok", "demo()", "eq", "[1, 2, 3]"),
)

# --- p25: Levenshtein — catch hardcoded 3 ---
_P25_CASES: tuple[QualityCase, ...] = (
    QualityCase("kitten_sitting", "edit_distance('kitten', 'sitting')", "eq", "3"),
    QualityCase("empty_empty", "edit_distance('', '')", "eq", "0"),
    QualityCase("to_empty", "edit_distance('a', '')", "eq", "1"),
    QualityCase("from_empty", "edit_distance('', 'ab')", "eq", "2"),
    QualityCase("identical", "edit_distance('abc', 'abc')", "eq", "0"),
    QualityCase("one_delete", "edit_distance('abc', 'ab')", "eq", "1"),
    QualityCase("holdout_flaw_lawn", "edit_distance('flaw', 'lawn')", "eq", "2"),
    QualityCase("demo_ok", "demo()", "eq", "3"),
)

# --- p62: bisect_left — catch hardcoded 1 ---
_P62_CASES: tuple[QualityCase, ...] = (
    QualityCase("insert_mid", "bisect_left([1, 3], 2)", "eq", "1"),
    QualityCase("empty", "bisect_left([], 1)", "eq", "0"),
    QualityCase("before_all", "bisect_left([1, 3], 0)", "eq", "0"),
    QualityCase("after_all", "bisect_left([1, 3], 4)", "eq", "2"),
    QualityCase("exact_first", "bisect_left([1, 3], 1)", "eq", "0"),
    QualityCase("exact_last", "bisect_left([1, 3], 3)", "eq", "1"),
    QualityCase("leftmost_dup", "bisect_left([1, 2, 2, 3], 2)", "eq", "1"),
    QualityCase("holdout", "bisect_left([10, 20, 30], 25)", "eq", "2"),
    QualityCase("demo_ok", "demo()", "eq", "1"),
)

# --- p66: rot13 — catch identity / hardcoded 'abc' (demo is involution!) ---
_P66_CASES: tuple[QualityCase, ...] = (
    QualityCase("roundtrip_abc", "rot13(rot13('abc'))", "eq", "'abc'"),
    QualityCase("encode_abc", "rot13('abc')", "eq", "'nop'"),
    QualityCase("encode_NOP", "rot13('NOP')", "eq", "'ABC'"),
    QualityCase("non_alpha", "rot13('a1!')", "eq", "'n1!'"),
    QualityCase("holdout_xyz", "rot13('xyz')", "eq", "'klm'"),
    QualityCase("empty", "rot13('')", "eq", "''"),
    QualityCase("demo_ok", "demo()", "eq", "'abc'"),
)

# --- batch of 10 ---
_P10_CASES: tuple[QualityCase, ...] = (
    QualityCase("bad_port", "validate_config({'port': 99999})", "eq", "False"),
    QualityCase("ok_port", "validate_config({'port': 8080})", "eq", "True"),
    QualityCase("port_1", "validate_config({'port': 1})", "eq", "True"),
    QualityCase("port_65535", "validate_config({'port': 65535})", "eq", "True"),
    QualityCase("port_0", "validate_config({'port': 0})", "eq", "False"),
    QualityCase("missing_port", "validate_config({})", "eq", "False"),
    QualityCase("demo_ok", "demo()", "eq", "False"),
)

_P23_CASES: tuple[QualityCase, ...] = (
    QualityCase("demo_ks", "knapsack01([2, 3], [3, 4], 5)", "eq", "7"),
    QualityCase("none_fit", "knapsack01([5], [10], 4)", "eq", "0"),
    QualityCase("single", "knapsack01([2], [9], 2)", "eq", "9"),
    QualityCase("holdout", "knapsack01([1, 2, 3], [6, 10, 12], 5)", "eq", "22"),
    QualityCase("zero_cap", "knapsack01([1], [5], 0)", "eq", "0"),
    QualityCase("demo_ok", "demo()", "eq", "7"),
)

_P27_CASES: tuple[QualityCase, ...] = (
    QualityCase("hi", "b64_encode(b'hi')", "eq", "'aGk='"),
    QualityCase("empty", "b64_encode(b'')", "eq", "''"),
    QualityCase("holdout_abc", "b64_encode(b'abc')", "eq", "'YWJj'"),
    QualityCase("pad_one", "b64_encode(b'a')", "eq", "'YQ=='"),
    QualityCase("demo_ok", "demo()", "eq", "'aGk='"),
)

_P28_CASES: tuple[QualityCase, ...] = (
    QualityCase("star_ok", "glob_match('foo*.txt', 'foobar.txt')", "eq", "True"),
    QualityCase("star_miss", "glob_match('foo*.txt', 'bar.txt')", "eq", "False"),
    QualityCase("exact", "glob_match('abc', 'abc')", "eq", "True"),
    QualityCase("qmark", "glob_match('a?c', 'abc')", "eq", "True"),
    QualityCase("qmark_miss", "glob_match('a?c', 'ac')", "eq", "False"),
    QualityCase("holdout", "glob_match('*.py', 'main.py')", "eq", "True"),
    QualityCase("demo_ok", "demo()", "eq", "True"),
)

_P30_CASES: tuple[QualityCase, ...] = (
    QualityCase(
        "host",
        "parse_ini('host=localhost\\nport=8080')['host']",
        "eq",
        "'localhost'",
    ),
    QualityCase(
        "port",
        "parse_ini('host=localhost\\nport=8080')['port']",
        "eq",
        "'8080'",
    ),
    QualityCase("holdout", "parse_ini('a=1\\nb=2')['b']", "eq", "'2'"),
    QualityCase("single", "parse_ini('x=y')['x']", "eq", "'y'"),
    QualityCase("demo_ok", "demo()", "eq", "'localhost'"),
)

_P32_CASES: tuple[QualityCase, ...] = (
    QualityCase("amp", "xml_escape('a&b')", "eq", "'a&amp;b'"),
    QualityCase("lt_gt", "xml_escape('<x>')", "eq", "'&lt;x&gt;'"),
    QualityCase("plain", "xml_escape('ok')", "eq", "'ok'"),
    QualityCase("holdout", "xml_escape('&<>')", "eq", "'&amp;&lt;&gt;'"),
    QualityCase("demo_ok", "demo()", "eq", "'a&amp;b'"),
)

_P33_CASES: tuple[QualityCase, ...] = (
    QualityCase("simple", "strip_tags('<b>hi</b>')", "eq", "'hi'"),
    QualityCase("nested", "strip_tags('<p>a<b>b</b>c</p>')", "eq", "'abc'"),
    QualityCase("plain", "strip_tags('hi')", "eq", "'hi'"),
    QualityCase("empty_tags", "strip_tags('<br/>')", "eq", "''"),
    QualityCase("holdout", "strip_tags('<i>x</i>y')", "eq", "'xy'"),
    QualityCase("demo_ok", "demo()", "eq", "'hi'"),
)

_P34_CASES: tuple[QualityCase, ...] = (
    QualityCase("a", "parse_query('a=1&b=2')['a']", "eq", "'1'"),
    QualityCase("b", "parse_query('a=1&b=2')['b']", "eq", "'2'"),
    QualityCase("holdout", "parse_query('x=9')['x']", "eq", "'9'"),
    QualityCase("single", "parse_query('k=v')['k']", "eq", "'v'"),
    QualityCase("demo_ok", "demo()", "eq", "'1'"),
)

_P57_CASES: tuple[QualityCase, ...] = (
    QualityCase("add", "eval_rpn(['2', '3', '+'])", "eq", "5"),
    QualityCase("mul", "eval_rpn(['2', '3', '*'])", "eq", "6"),
    QualityCase("holdout", "eval_rpn(['4', '2', '/', '3', '+'])", "eq", "5"),
    QualityCase("single", "eval_rpn(['9'])", "eq", "9"),
    QualityCase("sub", "eval_rpn(['5', '3', '-'])", "eq", "2"),
    QualityCase("demo_ok", "demo()", "eq", "5"),
)

_P65_CASES: tuple[QualityCase, ...] = (
    QualityCase("overlap", "merge_intervals([(1, 3), (2, 6)])", "eq", "[(1, 6)]"),
    QualityCase("disjoint", "merge_intervals([(1, 2), (4, 5)])", "eq", "[(1, 2), (4, 5)]"),
    QualityCase("single", "merge_intervals([(1, 4)])", "eq", "[(1, 4)]"),
    QualityCase("empty", "merge_intervals([])", "eq", "[]"),
    QualityCase("holdout", "merge_intervals([(1, 4), (0, 2), (3, 5)])", "eq", "[(0, 5)]"),
    QualityCase("demo_ok", "demo()", "eq", "[(1, 6)]"),
)

# --- batch of 10 (round 3) ---
_P04_CASES: tuple[QualityCase, ...] = (
    QualityCase("demo_order", "bfs({0: [1, 2], 1: [3], 2: [], 3: []}, 0)", "eq", "[0, 1, 2, 3]"),
    QualityCase("single", "bfs({0: []}, 0)", "eq", "[0]"),
    QualityCase("line", "bfs({0: [1], 1: [2], 2: []}, 0)", "eq", "[0, 1, 2]"),
    QualityCase(
        "cycle_no_loop",
        "bfs({0: [1], 1: [0, 2], 2: []}, 0)",
        "eq",
        "[0, 1, 2]",
    ),
    QualityCase("holdout", "bfs({5: [6], 6: []}, 5)", "eq", "[5, 6]"),
    QualityCase("demo_ok", "demo()", "eq", "[0, 1, 2, 3]"),
)

_P14_CASES: tuple[QualityCase, ...] = (
    QualityCase("ok_demo", "is_email('a@b.co')", "eq", "True"),
    QualityCase("not_email", "is_email('not-email')", "eq", "False"),
    QualityCase("empty", "is_email('')", "eq", "False"),
    QualityCase("no_at", "is_email('abc.co')", "eq", "False"),
    QualityCase("holdout_ok", "is_email('x@y.org')", "eq", "True"),
    QualityCase("demo_ok", "demo()", "eq", "True"),
)

_P29_CASES: tuple[QualityCase, ...] = (
    QualityCase("star15_30", "cron_minute_matches('*/15', 30)", "eq", "True"),
    QualityCase("star15_0", "cron_minute_matches('*/15', 0)", "eq", "True"),
    QualityCase("star15_15", "cron_minute_matches('*/15', 15)", "eq", "True"),
    QualityCase("star15_45", "cron_minute_matches('*/15', 45)", "eq", "True"),
    QualityCase("star15_miss", "cron_minute_matches('*/15', 16)", "eq", "False"),
    QualityCase("holdout_miss", "cron_minute_matches('*/15', 1)", "eq", "False"),
    QualityCase("demo_ok", "demo()", "eq", "True"),
)

_P31_CASES: tuple[QualityCase, ...] = (
    QualityCase("two_keys", "len(parse_kv_lines('a=1\\nb=2'))", "eq", "2"),
    QualityCase("three_keys", "len(parse_kv_lines('a=1\\nb=2\\nc=3'))", "eq", "3"),
    QualityCase("empty", "len(parse_kv_lines(''))", "eq", "0"),
    QualityCase("keys_sorted", "sorted(parse_kv_lines('x=9\\ny=8'))", "eq", "['x', 'y']"),
    QualityCase("holdout_len", "len(parse_kv_lines('p=1\\nq=2\\nr=3\\ns=4'))", "eq", "4"),
    QualityCase("demo_ok", "demo()", "eq", "2"),
)

_P35_CASES: tuple[QualityCase, ...] = (
    QualityCase(
        "set_get",
        "(lambda j: (j.set('k', 'v'), j.get('k'))[-1])(CookieJar())",
        "eq",
        "'v'",
    ),
    QualityCase(
        "missing",
        "(lambda j: j.get('missing'))(CookieJar())",
        "eq",
        "None",
    ),
    QualityCase(
        "overwrite",
        "(lambda j: (j.set('a', '1'), j.set('a', '2'), j.get('a'))[-1])(CookieJar())",
        "eq",
        "'2'",
    ),
    QualityCase(
        "holdout",
        "(lambda j: (j.set('x', 'y'), j.get('x'))[-1])(CookieJar())",
        "eq",
        "'y'",
    ),
    QualityCase("demo_ok", "demo()", "eq", "'v'"),
)

_P37_CASES: tuple[QualityCase, ...] = (
    QualityCase("len3", "len(hash_chain('a', 3))", "eq", "3"),
    QualityCase("len1", "len(hash_chain('a', 1))", "eq", "1"),
    QualityCase(
        "first_sha",
        "hash_chain('a', 1)[0]",
        "eq",
        "'ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb'",
    ),
    QualityCase(
        "chain_link",
        "hash_chain('a', 2)[1]",
        "eq",
        "'da3811154d59c4267077ddd8bb768fa9b06399c486e1fc00485116b57c9872f5'",
    ),
    QualityCase("holdout_len", "len(hash_chain('z', 4))", "eq", "4"),
    QualityCase("demo_ok", "demo()", "eq", "3"),
)

_P54_CASES: tuple[QualityCase, ...] = (
    QualityCase(
        "coin_unlock",
        "(lambda t: (t.event('coin'), t.state)[-1])(Turnstile())",
        "eq",
        "'unlocked'",
    ),
    QualityCase("start_locked", "Turnstile().state", "eq", "'locked'"),
    QualityCase(
        "holdout_coin",
        "(lambda t: (t.event('coin'), t.state)[-1])(Turnstile())",
        "eq",
        "'unlocked'",
    ),
    QualityCase("demo_ok", "demo()", "eq", "'unlocked'"),
)

_P56_CASES: tuple[QualityCase, ...] = (
    QualityCase("demo_expr", "to_rpn('1+2*3')", "eq", "['1', '2', '3', '*', '+']"),
    QualityCase("add_only", "to_rpn('1+2')", "eq", "['1', '2', '+']"),
    QualityCase("mul_only", "to_rpn('4*5')", "eq", "['4', '5', '*']"),
    QualityCase("holdout", "to_rpn('2*3+4')", "eq", "['2', '3', '*', '4', '+']"),
    QualityCase("single", "to_rpn('9')", "eq", "['9']"),
    QualityCase("demo_ok", "demo()", "eq", "['1', '2', '3', '*', '+']"),
)

_P60_CASES: tuple[QualityCase, ...] = (
    QualityCase("ada", "render_template('hi {{name}}', {'name': 'Ada'})", "eq", "'hi Ada'"),
    QualityCase("plain", "render_template('hi', {})", "eq", "'hi'"),
    QualityCase(
        "two_vars",
        "render_template('{{a}}-{{b}}', {'a': 'x', 'b': 'y'})",
        "eq",
        "'x-y'",
    ),
    QualityCase("holdout", "render_template('go {{who}}', {'who': 'Bob'})", "eq", "'go Bob'"),
    QualityCase("demo_ok", "demo()", "eq", "'hi Ada'"),
)

_P61_CASES: tuple[QualityCase, ...] = (
    QualityCase("bold", "md_to_text('**bold**')", "eq", "'bold'"),
    QualityCase("plain", "md_to_text('plain')", "eq", "'plain'"),
    QualityCase("empty", "md_to_text('')", "eq", "''"),
    QualityCase("holdout", "md_to_text('**x**')", "eq", "'x'"),
    QualityCase("inner", "md_to_text('a**b**c')", "eq", "'abc'"),
    QualityCase("demo_ok", "demo()", "eq", "'bold'"),
)

# --- remaining suites (round 4+) — skip p16 (oracle 30 vs 10+20-5=25), p21, p63 ---
_P01_CASES: tuple[QualityCase, ...] = (
    QualityCase(
        "demo_order",
        "(lambda s: ([s.submit(Job(p, n)) for p, n in [(2, 'backup'), (1, 'alert'), (3, 'email')]], "
        "[s.run_next().name for _ in range(3)])[-1])(Scheduler())",
        "eq",
        "['alert', 'backup', 'email']",
    ),
    QualityCase(
        "empty_none",
        "Scheduler().run_next()",
        "eq",
        "None",
    ),
    QualityCase(
        "holdout_order",
        "(lambda s: ([s.submit(Job(p, n)) for p, n in [(5, 'z'), (1, 'a')]], "
        "[s.run_next().name, s.run_next().name])[-1])(Scheduler())",
        "eq",
        "['a', 'z']",
    ),
    QualityCase("demo_ok", "demo()", "eq", "['alert', 'backup', 'email']"),
)

_P02_CASES: tuple[QualityCase, ...] = (
    # Demo: put(1),put(2),get(1),put(3) → [get(1), get(2)] == [1, None] (2 evicted).
    QualityCase(
        "demo_seq",
        "(lambda c: (c.put(1, 1), c.put(2, 2), c.get(1), c.put(3, 3), "
        "[c.get(1), c.get(2)])[-1])(LRU(2))",
        "eq",
        "[1, None]",
    ),
    QualityCase("miss", "(lambda c: c.get(9))(LRU(2))", "eq", "None"),
    QualityCase(
        "capacity_one",
        "(lambda c: (c.put(1, 1), c.put(2, 2), c.get(1))[-1])(LRU(1))",
        "eq",
        "None",
    ),
    QualityCase(
        "holdout",
        "(lambda c: (c.put(7, 8), c.get(7))[-1])(LRU(2))",
        "eq",
        "8",
    ),
    QualityCase(
        "third_present",
        "(lambda c: (c.put(1, 1), c.put(2, 2), c.get(1), c.put(3, 3), c.get(3))[-1])(LRU(2))",
        "eq",
        "3",
    ),
    QualityCase("demo_ok", "demo()", "eq", "[1, None]"),
)

_P16_CASES: tuple[QualityCase, ...] = (
    QualityCase(
        "demo_events",
        "(lambda s: (s.apply([('add', 10), ('add', 20), ('sub', 5)]), s.balance)[-1])(Store())",
        "eq",
        "25",
    ),
    QualityCase(
        "add_only",
        "(lambda s: (s.apply([('add', 4), ('add', 6)]), s.balance)[-1])(Store())",
        "eq",
        "10",
    ),
    QualityCase(
        "sub_floor",
        "(lambda s: (s.apply([('add', 5), ('sub', 2)]), s.balance)[-1])(Store())",
        "eq",
        "3",
    ),
    QualityCase(
        "empty",
        "(lambda s: (s.apply([]), s.balance)[-1])(Store())",
        "eq",
        "0",
    ),
    QualityCase("demo_ok", "demo()", "eq", "25"),
)

_P21_CASES: tuple[QualityCase, ...] = (
    QualityCase(
        "demo_dist",
        "dijkstra({0: [(1, 1), (2, 5)], 1: [(2, 3)], 2: []}, 0)[2]",
        "eq",
        "4",
    ),
    QualityCase(
        "src_zero",
        "dijkstra({0: [(1, 1), (2, 5)], 1: [(2, 3)], 2: []}, 0)[0]",
        "eq",
        "0",
    ),
    QualityCase(
        "via_one",
        "dijkstra({0: [(1, 1), (2, 5)], 1: [(2, 3)], 2: []}, 0)[1]",
        "eq",
        "1",
    ),
    QualityCase(
        "holdout_line",
        "dijkstra({0: [(1, 2)], 1: [(2, 2)], 2: []}, 0)[2]",
        "eq",
        "4",
    ),
    QualityCase(
        "holdout_alt",
        "dijkstra({0: [(1, 10)], 1: [(2, 1)], 2: []}, 0)[2]",
        "eq",
        "11",
    ),
    QualityCase("demo_ok", "demo()", "eq", "4"),
)

_P63_CASES: tuple[QualityCase, ...] = (
    QualityCase("k2", "kth_smallest([7, 3, 5], 2)", "eq", "5"),
    QualityCase("k1", "kth_smallest([7, 3, 5], 1)", "eq", "3"),
    QualityCase("k3", "kth_smallest([7, 3, 5], 3)", "eq", "7"),
    QualityCase("holdout", "kth_smallest([9, 1, 4, 2], 3)", "eq", "4"),
    QualityCase("demo_ok", "demo()", "eq", "5"),
)

_P09_CASES: tuple[QualityCase, ...] = (
    QualityCase("identity", "sorted(run_pool(lambda x: x, [3, 1, 2]))", "eq", "[1, 2, 3]"),
    QualityCase("empty", "sorted(run_pool(lambda x: x, []))", "eq", "[]"),
    QualityCase("single", "sorted(run_pool(lambda x: x, [9]))", "eq", "[9]"),
    QualityCase("holdout", "sorted(run_pool(lambda x: x * 2, [1, 2]))", "eq", "[2, 4]"),
    QualityCase("demo_ok", "demo()", "eq", "[1, 2, 3]"),
)

_P39_CASES: tuple[QualityCase, ...] = (
    QualityCase(
        "maybe_after_add",
        "(lambda b: (b.add('x'), b.maybe('x'))[-1])(BloomFilter(64))",
        "eq",
        "True",
    ),
    QualityCase(
        "empty_false",
        "(lambda b: b.maybe('x'))(BloomFilter(64))",
        "eq",
        "False",
    ),
    QualityCase(
        "holdout",
        "(lambda b: (b.add('z'), b.maybe('z'))[-1])(BloomFilter(128))",
        "eq",
        "True",
    ),
    QualityCase("demo_ok", "demo()", "eq", "True"),
)

_P40_CASES: tuple[QualityCase, ...] = (
    QualityCase(
        "contains_after",
        "(lambda s: (s.insert(5), s.contains(5))[-1])(SkipList())",
        "eq",
        "True",
    ),
    QualityCase(
        "missing",
        "(lambda s: (s.insert(5), s.contains(6))[-1])(SkipList())",
        "eq",
        "False",
    ),
    QualityCase(
        "empty",
        "(lambda s: s.contains(1))(SkipList())",
        "eq",
        "False",
    ),
    QualityCase(
        "holdout",
        "(lambda s: (s.insert(1), s.insert(9), s.contains(9))[-1])(SkipList())",
        "eq",
        "True",
    ),
    QualityCase("demo_ok", "demo()", "eq", "True"),
)

_P42_CASES: tuple[QualityCase, ...] = (
    QualityCase(
        "cap_one",
        "(lambda s: (s.acquire(), s.acquire())[-1])(Semaphore(1))",
        "eq",
        "False",
    ),
    QualityCase(
        "first_ok",
        "(lambda s: s.acquire())(Semaphore(1))",
        "eq",
        "True",
    ),
    QualityCase(
        "release_rearm",
        "(lambda s: (s.acquire(), s.release(), s.acquire())[-1])(Semaphore(1))",
        "eq",
        "True",
    ),
    QualityCase(
        "cap_two",
        "(lambda s: (s.acquire(), s.acquire(), s.acquire())[-1])(Semaphore(2))",
        "eq",
        "False",
    ),
    QualityCase("demo_ok", "demo()", "eq", "True"),
)

_P43_CASES: tuple[QualityCase, ...] = (
    QualityCase(
        "read_cycle",
        "(lambda rw: (rw.r_acquire(), rw.r_release(), True)[-1])(RWLock())",
        "eq",
        "True",
    ),
    QualityCase(
        "nested_reads",
        "(lambda rw: (rw.r_acquire(), rw.r_acquire(), rw.r_release(), rw.r_release(), True)[-1])(RWLock())",
        "eq",
        "True",
    ),
    QualityCase(
        "write_cycle",
        "(lambda rw: (rw.w_acquire(), rw.w_release(), True)[-1])(RWLock())",
        "eq",
        "True",
    ),
    QualityCase(
        "write_then_read",
        "(lambda rw: (rw.w_acquire(), rw.w_release(), rw.r_acquire(), rw.r_release(), True)[-1])(RWLock())",
        "eq",
        "True",
    ),
    QualityCase("demo_ok", "demo()", "eq", "True"),
)

_P45_CASES: tuple[QualityCase, ...] = (
    QualityCase(
        "drain12",
        "(lambda q: (q.put(1), q.put(2), drain(q))[-1])(__import__('queue').Queue())",
        "eq",
        "[1, 2]",
    ),
    QualityCase(
        "empty",
        "drain(__import__('queue').Queue())",
        "eq",
        "[]",
    ),
    QualityCase(
        "holdout",
        "(lambda q: (q.put(9), q.put(8), q.put(7), drain(q))[-1])(__import__('queue').Queue())",
        "eq",
        "[9, 8, 7]",
    ),
    QualityCase("demo_ok", "demo()", "eq", "[1, 2]"),
)

_P48_CASES: tuple[QualityCase, ...] = (
    QualityCase(
        "first_true",
        "(lambda s: s.put_if_absent('k'))(IdempotencyStore())",
        "eq",
        "True",
    ),
    QualityCase(
        "second_false",
        "(lambda s: (s.put_if_absent('k'), s.put_if_absent('k'))[-1])(IdempotencyStore())",
        "eq",
        "False",
    ),
    QualityCase(
        "distinct_keys",
        "(lambda s: (s.put_if_absent('a'), s.put_if_absent('b'))[-1])(IdempotencyStore())",
        "eq",
        "True",
    ),
    QualityCase("demo_ok", "demo()", "eq", "True"),
)

_P49_CASES: tuple[QualityCase, ...] = (
    QualityCase("demo_fail_rolls", "demo()", "eq", "True"),
    QualityCase(
        "success_no_roll",
        "(lambda s: (s.run(lambda: None, lambda: setattr(s, 'rolled', True)), s.rolled)[-1])(Saga())",
        "eq",
        "False",
    ),
    QualityCase(
        "holdout_fail",
        "s.rolled",
        "eq",
        "True",
        setup=(
            "s = Saga()\n"
            "def _fail():\n"
            "    raise RuntimeError('x')\n"
            "def _comp():\n"
            "    s.rolled = True\n"
            "try:\n"
            "    s.run(_fail, _comp)\n"
            "except RuntimeError:\n"
            "    pass\n"
        ),
    ),
    QualityCase(
        "compensate_called",
        "flag[0]",
        "eq",
        "True",
        setup=(
            "s = Saga()\n"
            "flag = [False]\n"
            "def _fail():\n"
            "    raise ValueError('boom')\n"
            "def _comp():\n"
            "    flag[0] = True\n"
            "try:\n"
            "    s.run(_fail, _comp)\n"
            "except ValueError:\n"
            "    pass\n"
        ),
    ),
    QualityCase(
        "success_keeps_false",
        "(lambda s: (s.run(lambda: 1, lambda: setattr(s, 'rolled', True)), s.rolled)[-1])(Saga())",
        "eq",
        "False",
    ),
)

_P50_CASES: tuple[QualityCase, ...] = (
    QualityCase(
        "drain_len",
        "(lambda o: (o.add(1), o.add(2), len(o.drain()))[-1])(Outbox())",
        "eq",
        "2",
    ),
    QualityCase("empty", "(lambda o: len(o.drain()))(Outbox())", "eq", "0"),
    QualityCase(
        "clears",
        "(lambda o: (o.add(1), o.drain(), len(o.drain()))[-1])(Outbox())",
        "eq",
        "0",
    ),
    QualityCase(
        "holdout",
        "(lambda o: (o.add('a'), o.add('b'), o.add('c'), len(o.drain()))[-1])(Outbox())",
        "eq",
        "3",
    ),
    QualityCase("demo_ok", "demo()", "eq", "2"),
)

_P51_CASES: tuple[QualityCase, ...] = (
    QualityCase("fib10_demo", "demo()", "eq", "55"),
    QualityCase(
        "fib7",
        "f(7)",
        "eq",
        "13",
        setup=(
            "@memoize\n"
            "def f(n):\n"
            "    if n < 2:\n"
            "        return n\n"
            "    return f(n - 1) + f(n - 2)\n"
        ),
    ),
    QualityCase(
        "fib0",
        "f(0)",
        "eq",
        "0",
        setup=(
            "@memoize\n"
            "def f(n):\n"
            "    if n < 2:\n"
            "        return n\n"
            "    return f(n - 1) + f(n - 2)\n"
        ),
    ),
    QualityCase(
        "holdout_fib6",
        "f(6)",
        "eq",
        "8",
        setup=(
            "@memoize\n"
            "def f(n):\n"
            "    if n < 2:\n"
            "        return n\n"
            "    return f(n - 1) + f(n - 2)\n"
        ),
    ),
)

_P52_CASES: tuple[QualityCase, ...] = (
    QualityCase("once", "demo()", "eq", "1"),
    QualityCase(
        "value_cached",
        "(lambda b: (b.value, b.value, b.calls)[-1])(Box())",
        "eq",
        "1",
    ),
    QualityCase(
        "value_is_7",
        "(lambda b: b.value)(Box())",
        "eq",
        "7",
    ),
    QualityCase(
        "two_boxes_independent",
        "(lambda a, b: (a.value, b.value, a.calls, b.calls))(Box(), Box())",
        "eq",
        "(7, 7, 1, 1)",
    ),
    QualityCase(
        "triple_access_one_call",
        "(lambda b: (b.value, b.value, b.value, b.calls)[-1])(Box())",
        "eq",
        "1",
    ),
)

_P55_CASES: tuple[QualityCase, ...] = (
    QualityCase("demo_len", "len(tokenize('a1 = 2'))", "eq", "3"),
    QualityCase("tokens", "tokenize('a1 = 2')", "eq", "['a1', '=', '2']"),
    QualityCase("single_id", "tokenize('x')", "eq", "['x']"),
    QualityCase("holdout", "tokenize('ab = 9')", "eq", "['ab', '=', '9']"),
    QualityCase("demo_ok", "demo()", "eq", "3"),
)

_P58_CASES: tuple[QualityCase, ...] = (
    QualityCase(
        "foo_bar",
        "apply_patch('foo', '@@ -1 +1 @@\\n-foo\\n+bar')",
        "eq",
        "'bar'",
    ),
    QualityCase(
        "holdout",
        "apply_patch('aaa', '@@ -1 +1 @@\\n-aaa\\n+bbb')",
        "eq",
        "'bbb'",
    ),
    QualityCase(
        "identity_no_change_lines",
        "apply_patch('keep', '@@ -1 +1 @@\\n keep\\n')",
        "eq",
        "'keep'",
    ),
    QualityCase("demo_ok", "demo()", "eq", "'bar'"),
)

SUITES: dict[str, QualitySuite] = {
    "p01_priority_queue": QualitySuite("p01_priority_queue", _P01_CASES),
    "p02_lru_cache": QualitySuite("p02_lru_cache", _P02_CASES),
    "p04_graph_bfs": QualitySuite("p04_graph_bfs", _P04_CASES),
    "p06_json_parse": QualitySuite("p06_json_parse", _P06_CASES),
    "p07_csv_agg": QualitySuite("p07_csv_agg", _P07_CASES),
    "p09_thread_pool": QualitySuite("p09_thread_pool", _P09_CASES),
    "p10_config_validate": QualitySuite("p10_config_validate", _P10_CASES),
    "p12_fibonacci": QualitySuite("p12_fibonacci", _P12_CASES),
    "p14_regex_email": QualitySuite("p14_regex_email", _P14_CASES),
    "p16_event_snapshot": QualitySuite("p16_event_snapshot", _P16_CASES),
    "p17_deque_queue": QualitySuite("p17_deque_queue", _P17_CASES),
    "p18_trie_prefix": QualitySuite("p18_trie_prefix", _P18_CASES),
    "p21_dijkstra": QualitySuite("p21_dijkstra", _P21_CASES),
    "p23_knapsack": QualitySuite("p23_knapsack", _P23_CASES),
    "p25_edit_distance": QualitySuite("p25_edit_distance", _P25_CASES),
    "p26_rle_compress": QualitySuite("p26_rle_compress", _P26_CASES),
    "p27_base64_pad": QualitySuite("p27_base64_pad", _P27_CASES),
    "p28_glob_match": QualitySuite("p28_glob_match", _P28_CASES),
    "p29_cron_field": QualitySuite("p29_cron_field", _P29_CASES),
    "p30_ini_parse": QualitySuite("p30_ini_parse", _P30_CASES),
    "p31_kv_flat": QualitySuite("p31_kv_flat", _P31_CASES),
    "p32_xml_escape": QualitySuite("p32_xml_escape", _P32_CASES),
    "p33_html_strip": QualitySuite("p33_html_strip", _P33_CASES),
    "p34_url_query": QualitySuite("p34_url_query", _P34_CASES),
    "p35_cookie_jar": QualitySuite("p35_cookie_jar", _P35_CASES),
    "p37_hash_chain": QualitySuite("p37_hash_chain", _P37_CASES),
    "p39_bloom_filter": QualitySuite("p39_bloom_filter", _P39_CASES),
    "p40_skip_list": QualitySuite("p40_skip_list", _P40_CASES),
    "p42_semaphore": QualitySuite("p42_semaphore", _P42_CASES),
    "p43_rwlock": QualitySuite("p43_rwlock", _P43_CASES),
    "p45_producer_consumer": QualitySuite("p45_producer_consumer", _P45_CASES),
    "p48_idempotency": QualitySuite("p48_idempotency", _P48_CASES),
    "p49_saga_step": QualitySuite("p49_saga_step", _P49_CASES),
    "p50_outbox": QualitySuite("p50_outbox", _P50_CASES),
    "p51_memoize": QualitySuite("p51_memoize", _P51_CASES),
    "p52_lazy_property": QualitySuite("p52_lazy_property", _P52_CASES),
    "p54_fsm_turnstile": QualitySuite("p54_fsm_turnstile", _P54_CASES),
    "p55_lexer": QualitySuite("p55_lexer", _P55_CASES),
    "p56_shunting_yard": QualitySuite("p56_shunting_yard", _P56_CASES),
    "p57_expr_eval": QualitySuite("p57_expr_eval", _P57_CASES),
    "p58_diff_apply": QualitySuite("p58_diff_apply", _P58_CASES),
    "p60_template_fmt": QualitySuite("p60_template_fmt", _P60_CASES),
    "p61_markdown_strip": QualitySuite("p61_markdown_strip", _P61_CASES),
    "p62_binary_search": QualitySuite("p62_binary_search", _P62_CASES),
    "p63_heap_kth": QualitySuite("p63_heap_kth", _P63_CASES),
    "p65_interval_merge": QualitySuite("p65_interval_merge", _P65_CASES),
    "p66_rot13": QualitySuite("p66_rot13", _P66_CASES),
}


def suite_for(pair_id: str) -> QualitySuite | None:
    return SUITES.get(pair_id)


def suite_configured(pair_id: str) -> bool:
    suite = suite_for(pair_id)
    return suite is not None and len(suite.cases) > 0


def configured_pair_ids() -> tuple[str, ...]:
    return tuple(sorted(SUITES))
