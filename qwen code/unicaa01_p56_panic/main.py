"""Shunting Yard Algorithm for RPN."""

def to_rpn(expr: str) -> list[str]:
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    stack = []
    output = []
    tokens = expr.split()

    for token in tokens:
        if token.isdigit():
            output.append(token)
        elif token in precedence:
            while stack and stack[-1] != '(' and precedence[token] <= precedence[stack[-1]]:
                output.append(stack.pop())
            stack.append(token)
        elif token == '(':
            stack.append(token)
        elif token == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()

    while stack:
        output.append(stack.pop())

    return output

def demo():
    return to_rpn("1+2*3")
