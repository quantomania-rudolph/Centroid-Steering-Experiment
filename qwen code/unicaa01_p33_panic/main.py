"""Broken HTML strip."""
import re


def strip_tags(html):
    # Use regex to remove HTML tags
    clean = re.sub(r'<[^>]*>', '', html)
    return clean

def demo():
    return strip_tags("<b>hi</b>")
