"""Hidden Markov Model forward algorithm — broken scaling."""

from __future__ import annotations


def forward_log(
    obs: list[int],
    start: list[float],
    trans: list[list[float]],
    emit: list[list[float]],
) -> float:
  """Log P(obs | model) via forward pass in log-space."""
  n_states = len(start)
  t_len = len(obs)
  # BUG: initializes with linear probs not log
  alpha = [start[i] * emit[i][obs[0]] for i in range(n_states)]
  for t in range(1, t_len):
    nxt = [0.0] * n_states
    for j in range(n_states):
      for i in range(n_states):
        nxt[j] += alpha[i] * trans[i][j] * emit[j][obs[t]]
    alpha = nxt
  return sum(alpha)


def demo() -> float:
  start = [0.6, 0.4]
  trans = [[0.7, 0.3], [0.4, 0.6]]
  emit = [[0.1, 0.9], [0.8, 0.2]]
  obs = [0, 1, 0]
  return forward_log(obs, start, trans, emit)


def main() -> None:
  p = demo()
  assert 0.01 < p < 0.5, p
  print(f"P(obs)={p:.6f}")


if __name__ == "__main__":
  main()
