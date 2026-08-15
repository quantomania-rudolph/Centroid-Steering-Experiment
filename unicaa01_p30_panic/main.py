def parse_ini(text: str) -> dict:
    ini_dict = {}
    lines = text.strip().split('\n')
    for line in lines:
        if '=' in line:
            key, value = line.split('=', 1)
            ini_dict[key.strip()] = value.strip()
    return ini_dict

def demo() -> dict:
    ini_text = "host=localhost"
    return parse_ini(ini_text)

if __name__ == "__main__":
    print(demo())
