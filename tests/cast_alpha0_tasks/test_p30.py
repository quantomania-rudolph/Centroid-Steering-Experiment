"""Pytest for healed p30 ini parser."""

from __future__ import annotations

import main


def test_demo_localhost() -> None:
    assert main.demo() == "localhost"


def test_parse_ini_ignores_comments() -> None:
    parsed = main.parse_ini("# comment\nhost=127.0.0.1\nport=9")
    assert parsed["host"] == "127.0.0.1"
    assert parsed["port"] == "9"


def test_parse_ini_skips_empty_lines() -> None:
    parsed = main.parse_ini("\n\nhost=alpha\n\nport=42\n")
    assert parsed["host"] == "alpha"
    assert parsed["port"] == "42"


def test_parse_ini_last_wins_duplicate_keys() -> None:
    parsed = main.parse_ini("port=1\nport=8080")
    assert parsed["port"] == "8080"
