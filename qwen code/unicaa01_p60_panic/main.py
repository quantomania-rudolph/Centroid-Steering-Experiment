def render_template(tpl: str, ctx: dict) -> str:
    for key, value in ctx.items():
        tpl = tpl.replace(f"{{{key}}}", value)
    return tpl

def demo():
    return render_template("hi {{name}}", {"name": "Ada"})
