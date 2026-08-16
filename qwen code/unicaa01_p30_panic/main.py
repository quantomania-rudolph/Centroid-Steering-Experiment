"""Broken INI."""

def parse_ini(text):
    result = {}
    for line in text.split('\n'):
        if '=' in line:
            key, value = line.split('=', 1)
            result[key.strip()] = value.strip()
    return result


def demo():
    return parse_ini("host=localhost\nport=8080")["host"]
