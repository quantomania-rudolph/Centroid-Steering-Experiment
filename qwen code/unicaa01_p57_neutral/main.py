"""RPN eval kata."""
def eval_rpn(tokens: list[str]) -> float:
    stack = []
    for token in tokens:
        if token.isdigit():
            stack.append(int(token))
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
    return stack[0]

def demo():
    result = eval_rpn(['2', '3', '+'])
    print(f"eval_rpn(['2', '3', '+']) == {result}")

if __name__ == '__main__':
    demo()
