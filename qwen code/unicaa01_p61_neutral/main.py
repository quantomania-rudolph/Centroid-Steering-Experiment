"""Markdown kata."""
import re


def md_to_text(md: str) -> str:
    # Strip markdown to plain text
    plain_text = re.sub(r'\*{2}(.*?)\*{2}', r'\1', md)  # Remove **bold**
    plain_text = re.sub(r'\*{1}(.*?)\*{1}', r'\1', plain_text)  # Remove *italic*
    plain_text = re.sub(r'_{2}(.*?)_{2}', r'\1', plain_text)  # Remove __underline__
    plain_text = re.sub(r'_{1}(.*?)_{1}', r'\1', plain_text)  # Remove _italic_
    plain_text = re.sub(r'`{3}(.*?)`{3}', r'\1', plain_text, flags=re.DOTALL)  # Remove ```code```
    plain_text = re.sub(r'`{1}(.*?)`{1}', r'\1', plain_text)  # Remove `code`
    plain_text = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1', plain_text)  # Remove [link](url)
    plain_text = re.sub(r'\n+', '\n', plain_text)  # Remove extra newlines
    plain_text = re.sub(r'\s+', ' ', plain_text)  # Remove extra spaces
    return plain_text.strip()

def demo():
    md = "**This is bold** and *this is italic* and __this is underline__ and _this is italic_ and `this is code` and [this is a link](http://example.com)"
    print(md_to_text(md))
