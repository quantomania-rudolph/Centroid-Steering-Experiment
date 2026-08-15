"""p06 uses stdlib json — fails SEM_invariants."""

import json


def parse_json(s: str):
    return json.loads(s)


def demo():
    return parse_json("[1,2]")
