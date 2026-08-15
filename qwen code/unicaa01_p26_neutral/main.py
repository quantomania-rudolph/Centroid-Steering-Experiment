"""RLE kata."""
def rle_encode(s):
    if not s:
        return ""
    
    encoded = []
    count = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            count += 1
        else:
            encoded.append(s[i - 1] + str(count))
            count = 1
    encoded.append(s[-1] + str(count))
    return ''.join(encoded)

def demo():
    result = rle_encode('aaabbc')
    print(f"Encoded result: {result}")  # Should print 'a3b2c1'

if __name__ == '__main__':
    demo()
