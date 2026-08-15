import re


def is_email(s):
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_regex, s) is not None

def demo():
    return is_email("a@b.co") and not is_email("not-email")
