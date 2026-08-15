"""Broken INI."""


def parse_ini(text):
    raise NotImplementedError("parse_ini not implemented")


def demo():
    return parse_ini("host=localhost\nport=8080")["host"]
