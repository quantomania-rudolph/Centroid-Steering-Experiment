def rle_encode(s):
    encoded = []
    count = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            count += 1
        else:
            encoded.append(f"{s[i - 1]}{count}")
            count = 1
    encoded.append(f"{s[-1]}{count}")
    return ''.join(encoded)

def demo():
    test_string = "aaabbc"
    expected_output = "a3b2c1"
    actual_output = rle_encode(test_string)
    return actual_output == expected_output
