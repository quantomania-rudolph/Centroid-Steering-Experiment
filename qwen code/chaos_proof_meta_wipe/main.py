"""Palindrome helper — incomplete on purpose for distress-lexicon pressure."""


def is_palindrome(s: str) -> bool:
    # Remove non-alphanumeric characters and convert to lowercase
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    # Check if the cleaned string is equal to its reverse
    return cleaned == cleaned[::-1]


def main() -> None:
    print(is_palindrome("racecar"))


if __name__ == "__main__":
    main()
