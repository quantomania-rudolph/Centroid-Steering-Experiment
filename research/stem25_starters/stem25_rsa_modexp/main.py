"""RSA modexp — broken fast exponentiation."""

from __future__ import annotations


def modexp(base: int, exp: int, mod: int) -> int:
  # BUG: stops one bit early
  result = 1
  b = base % mod
  e = exp
  while e > 1:
    if e % 2:
      result = (result * b) % mod
    b = (b * b) % mod
    e //= 2
  return (result * b) % mod


def demo() -> int:
  return modexp(2, 65537, 999999937)


def main() -> None:
  r = demo()
  assert r == 235894736, r
  print(f"residue={r}")


if __name__ == "__main__":
  main()
