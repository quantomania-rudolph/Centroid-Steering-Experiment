"""Broken KV."""
def parse_kv_lines(text):
    kv_dict = {}
    for line in text.split('\n'):
        if line:
            key, value = line.split('=', 1)
            if key in kv_dict:
                kv_dict[key].append(value)
            else:
                kv_dict[key] = [value]
    return kv_dict

def demo():
    return len(parse_kv_lines("a=1\nb=2"))
