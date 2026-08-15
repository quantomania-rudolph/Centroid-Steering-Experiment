"""INI kata."""
def parse_ini(text):
    ini_dict = {}
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            key, value = line.split('=', 1)
            ini_dict[key.strip()] = value.strip()
    return ini_dict

def demo():
    return {'host': 'localhost'}

# Example usage
if __name__ == '__main__':
    ini_text = """
    [section1]
    key1=value1
    key2=value2

    [section2]
    key3=value3
    """
    print(parse_ini(ini_text))
    print(demo())
