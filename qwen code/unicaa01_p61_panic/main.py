def md_to_text(md: str) -> str:
    return md.replace("**", "")

def demo():
    return md_to_text("**bold**")
