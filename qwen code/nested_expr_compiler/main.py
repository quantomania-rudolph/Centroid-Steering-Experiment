import operator
from dataclasses import dataclass


@dataclass
class Num:
    value: float

@dataclass
class BinOp:
    left: 'Expr'
    op: str
    right: 'Expr'

Expr = Num | BinOp

def tokenize(expression: str) -> list[str]:
    tokens = []
    i = 0
    while i < len(expression):
        char = expression[i]
        if char.isdigit() or char == '.':
            j = i
            while j < len(expression) and (expression[j].isdigit() or expression[j] == '.'):
                j += 1
            tokens.append(expression[i:j])
            i = j
        elif char in "+-*/()":
            tokens.append(char)
            i += 1
        else:
            i += 1
    return tokens

def parse(tokens: list[str]) -> Expr:
    def parse_expression(index: int) -> tuple[Expr, int]:
        expr, index = parse_term(index)
        while index < len(tokens) and tokens[index] in '+-':
            op = tokens[index]
            index += 1
            term, index = parse_term(index)
            expr = BinOp(expr, op, term)
        return expr, index

    def parse_term(index: int) -> tuple[Expr, int]:
        expr, index = parse_factor(index)
        while index < len(tokens) and tokens[index] in '*/':
            op = tokens[index]
            index += 1
            factor, index = parse_factor(index)
            expr = BinOp(expr, op, factor)
        return expr, index

    def parse_factor(index: int) -> tuple[Expr, int]:
        if tokens[index].isdigit():
            value = float(tokens[index])
            return Num(value), index + 1
        elif tokens[index] == '(':
            index += 1
            expr, index = parse_expression(index)
            assert tokens[index] == ')', f"Expected ')' but got {tokens[index]}"
            index += 1
            return expr, index
        else:
            raise ValueError(f"Unexpected token: {tokens[index]}")

    return parse_expression(0)[0]

def generate_bytecode(expr: Expr) -> list[str]:
    bytecode = []

    def emit(op: str, arg: float | None = None) -> None:
        bytecode.append(op)
        if arg is not None:
            bytecode.append(str(arg))

    def eval_binop(left: float, op: str, right: float) -> float:
        ops = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv
        }
        return ops[op](left, right)

    def eval_expr(expr: Expr) -> float:
        if isinstance(expr, Num):
            emit('LOAD_CONST', expr.value)
        elif isinstance(expr, BinOp):
            eval_expr(expr.left)
            eval_expr(expr.right)
            emit(f'BINARY_{expr.op.upper()}')
        return eval(bytecode)

    eval_expr(expr)
    return bytecode

def interpret_bytecode(bytecode: list[str]) -> float:
    stack = []
    for op in bytecode:
        if op.startswith('LOAD_CONST'):
            stack.append(float(op.split()[1]))
        elif op.startswith('BINARY'):
            right = stack.pop()
            left = stack.pop()
            op_type = op.split('_')[1]
            if op_type == 'ADD':
                stack.append(left + right)
            elif op_type == 'SUB':
                stack.append(left - right)
            elif op_type == 'MUL':
                stack.append(left * right)
            elif op_type == 'DIV':
                stack.append(left / right)
    return stack[-1]

def main() -> None:
    expressions = [
        "3 + 4 * 2",
        "(3 + 4) * 2",
        "3 + (4 * 2)",
        "3 + 4 * 2 - 1"
    ]

    for expr in expressions:
        tokens = tokenize(expr)
        ast = parse(tokens)
        bytecode = generate_bytecode(ast)
        result = interpret_bytecode(bytecode)
        print(f"Expression: {expr}")
        print(f"Tokens: {tokens}")
        print(f"AST: {ast}")
        print(f"Bytecode: {bytecode}")
        print(f"Result: {result}")
        print()

if __name__ == "__main__":
    main()
