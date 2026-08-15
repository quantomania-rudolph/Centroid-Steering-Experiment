def tokenize(src: str) -> list[str]:
    tokens = []
    i = 0
    while i < len(src):
        if src[i].isalpha():
            j = i + 1
            while j < len(src) and src[j].isalnum():
                j += 1
            tokens.append(src[i:j])
            i = j
        elif src[i].isdigit():
            j = i + 1
            while j < len(src) and src[j].isdigit():
                j += 1
            tokens.append(src[i:j])
            i = j
        elif src[i].isspace():
            i += 1
        else:
            raise ValueError(f"Unexpected character: {src[i]}")
    return tokens

def demo():
    return len(tokenize("a1 = 2"))
