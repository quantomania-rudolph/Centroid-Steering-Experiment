"""Task workspace for Aider + Qwen."""

import re


def is_palindrome(s: str) -> bool:
    # Remove non-alphanumeric characters and convert to lowercase
    cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()
    # Check if the cleaned string is equal to its reverse
    return cleaned == cleaned[::-1]

def main() -> None:
    test_strings = ['Race car', 'hello']
    for test_string in test_strings:
        result = is_palindrome(test_string)
        print(f"'{test_string}' is a palindrome: {result}")

if __name__ == "__main__":
    main()
