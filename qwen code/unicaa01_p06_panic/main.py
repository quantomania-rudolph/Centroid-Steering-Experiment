def parse_json(s):
    import json
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None

def demo():
    assert parse_json('[1,2]') == [1, 2]
