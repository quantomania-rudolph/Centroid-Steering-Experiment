"""Parse INI file."""


def parse_ini(text):
    result = {}
    for line in text.split('\n'):
        line = line.strip()
        if line and not line.startswith('#'):
            key, value = line.split('=')
            result[key.strip()] = value.strip()
    return result


def demo():
    return parse_ini("host=localhost\nport=8080")["host"]
