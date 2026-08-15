"""Broken semver."""
def semver_compare(a: str, b: str) -> int:
    raise NotImplementedError("semver_compare not implemented")

def demo():
    return semver_compare("1.2.0", "1.1.9") > 0
