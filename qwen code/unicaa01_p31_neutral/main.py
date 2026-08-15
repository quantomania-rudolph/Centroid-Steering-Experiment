"""KV kata."""
def parse_kv_lines(text):
    kv_dict = {}
    for line in text.splitlines():
        key, value = line.split('=')
        kv_dict[key.strip()] = value.strip()
    return kv_dict

def demo():
    sample_text = "key1=value1\nkey2=value2"
    result = parse_kv_lines(sample_text)
    return result

if __name__ == '__main__':
    print(demo())
