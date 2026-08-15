def parse_query(qs):
    """Parse a query string into a dictionary."""
    result = {}
    pairs = qs.split('&')
    for pair in pairs:
        key, value = pair.split('=')
        result[key] = value
    return result

def demo():
    return parse_query("a=1&b=2")["a"]
