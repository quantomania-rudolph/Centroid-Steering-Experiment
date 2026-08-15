"""Template kata."""
def render_template(tpl: str, ctx: dict) -> str:
    return tpl.format(**ctx)

def demo():
    return render_template('hi {{name}}', {'name': 'Ada'})
