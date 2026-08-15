"""Lexer kata."""
import re


def tokenize(src: str) -> list[str]:
    tokens = []
    # Regular expression to match IDs and integers
    pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b|\d+'
    
    for match in re.finditer(pattern, src):
        tokens.append(match.group())
    
    return tokens

def demo():
    src = 'a1 = 2'
    tokens = tokenize(src)
    print(tokens)

if __name__ == '__main__':
    demo()
