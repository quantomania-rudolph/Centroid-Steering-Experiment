def apply_patch(text: str, patch: str) -> str:
    lines = patch.split('\n')
    hunk = lines[0].split('@@')[1].strip()
    start_line, end_line = map(int, hunk.split(' ')[1].split(','))
    
    # Extract the lines to be replaced
    replace_lines = lines[1:end_line - start_line + 1]
    
    # Apply the patch
    result_lines = text.split('\n')
    result_lines[start_line - 1:start_line - 1 + len(replace_lines)] = replace_lines
    
    return '\n'.join(result_lines)

def demo():
    p = "@@ -1 +1 @@\n-foo\n+bar"
    return apply_patch("foo", p)
