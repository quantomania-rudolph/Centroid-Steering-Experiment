"""Mini expression compiler — intentionally broken for final review harness."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class Token:
    kind: str
    value: str


class Lexer:
  """Tokenize arithmetic + identifiers."""

  def __init__(self, src: str) -> None:
      self.src = src
      self.pos = 0

  def tokenize(self) -> list[Token]:
      # BUG: only handles digits, not identifiers or operators properly
      nums = re.findall(r"\d+", self.src)
      return [Token("NUM", n) for n in nums]


class Parser:
  def __init__(self, tokens: list[Token]) -> None:
      self.tokens = tokens
      self.i = 0

  def parse(self) -> dict[str, Any]:
      # BUG: returns flat list, not an AST
      return {"nums": [t.value for t in self.tokens if t.kind == "NUM"]}


def compile_expr(src: str) -> dict[str, Any]:
    lex = Lexer(src)
    return Parser(lex.tokenize()).parse()


def main() -> None:
    ast = compile_expr("x = 2 + 3 * 4")
    print(ast)


if __name__ == "__main__":
    main()
