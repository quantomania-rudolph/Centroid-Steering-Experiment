"""Diff apply kata."""

def apply_patch(text: str, patch: str) -> str:
    lines = text.split('\n')
    hunk_lines = patch.split('\n')
    
    # Parse the hunk header to get the start line and number of lines to replace
    hunk_header = hunk_lines[0]
    if not hunk_header.startswith('@@'):
        raise ValueError("Invalid hunk header")
    
    start_line, _num_lines = map(int, hunk_header[4:-1].split(','))
    
    # Apply the patch
    for i in range(1, len(hunk_lines)):
        if hunk_lines[i].startswith('-'):
            lines.pop(start_line - 1)
        elif hunk_lines[i].startswith('+'):
            lines.insert(start_line - 1, hunk_lines[i][1:])
            start_line += 1
    
    return '\n'.join(lines)

def demo():
    text = "foo\nbar\nbaz"
    patch = "@@ -1,1 +1,1 @@\n-foo\n+bar"
    result = apply_patch(text, patch)
    print(result)  # Output should be "bar\nbar\nbaz"

if __name__ == '__main__':
    demo()
