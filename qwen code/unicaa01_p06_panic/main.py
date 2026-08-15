import json


def parse_json(s):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None

def demo():
    assert parse_json('[1,2]') == [1, 2]
    print("Demo passed!")
