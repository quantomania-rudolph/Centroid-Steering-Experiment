"""Broken RPN eval."""
def eval_rpn(tokens: list[str]) -> float:
    stack = []
    for token in tokens:
        if token.isdigit():
            stack.append(float(token))
        elif token == '+':
            stack.append(stack.pop() + stack.pop())
        elif token == '-':
            b = stack.pop()
            a = stack.pop()
            stack.append(a - b)
        elif token == '*':
            stack.append(stack.pop() * stack.pop())
        elif token == '/':
            b = stack.pop()
            a = stack.pop()
            stack.append(a / b)
    return stack.pop()

def demo():
    return eval_rpn(["2", "3", "+"])
