"""Shunting yard kata."""
import re


def to_rpn(expr: str) -> list[str]:
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
    associativity = {'+': 'L', '-': 'L', '*': 'L', '/': 'L', '^': 'R'}
    output = []
    operators = []
    
    tokens = re.findall(r'\d+|\+|\-|\*|\/|\^|\(|\)', expr)
    
    for token in tokens:
        if token.isdigit():
            output.append(token)
        elif token == '(':
            operators.append(token)
        elif token == ')':
            while operators and operators[-1] != '(':
                output.append(operators.pop())
            operators.pop()
        else:
            while (operators and operators[-1] != '(' and
                   (precedence[token] < precedence[operators[-1]] or
                    (precedence[token] == precedence[operators[-1]] and associativity[token] == 'L'))):
                output.append(operators.pop())
            operators.append(token)
    
    while operators:
        output.append(operators.pop())
    
    return output

def demo():
    expr = '1+2*3'
    rpn = to_rpn(expr)
    print(f"Expression: {expr}")
    print(f"RPN: {rpn}")

if __name__ == '__main__':
    demo()
