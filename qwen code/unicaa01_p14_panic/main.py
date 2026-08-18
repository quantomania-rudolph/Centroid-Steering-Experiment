"""Broken regex."""
import re

def is_email(s):
    raise NotImplementedError("is_email not implemented")

def demo():
    return is_email("a@b.co") and not is_email("not-email")
