"""Suffix array kata."""
def suffix_array(s: str) -> list[int]:
    suffixes = [(s[i:], i) for i in range(len(s))]
    sorted_suffixes = sorted(suffixes)
    return [index for _, index in sorted_suffixes]

def demo():
    s = 'banana'
    result = suffix_array(s)
    return result[:4]

if __name__ == '__main__':
    print(demo())
