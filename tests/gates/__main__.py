"""Allow `python -m tests.gates.run_all --through N`."""

from tests.gates.run_all import main

if __name__ == "__main__":
    raise SystemExit(main())
