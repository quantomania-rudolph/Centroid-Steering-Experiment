"""Thin strip-split parser — must fail v4 quality suite (empty array + raises)."""


def parse_json(s):
    s = s[1:-1]
    return [int(x) for x in s.split(",")]


def demo():
    return parse_json("[1,2]")
