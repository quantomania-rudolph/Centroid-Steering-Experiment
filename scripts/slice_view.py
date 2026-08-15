#!/usr/bin/env python
"""Build layer x position slice visualization for J-Lens readouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        default="Count to five and introspect deeply about your reasoning. 1 2 3 4 5.",
    )
    parser.add_argument("--layers", default="11-22")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--lens",
        type=Path,
        default=ROOT / "artifacts" / "lenses" / "qwen_jacobian_lens.pt",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "artifacts" / "slices",
    )
    args = parser.parse_args()

    import torch

    from jlens_qwen.jacobian_fit import identity_lens
    from jlens_qwen.jacobian_lens import FittedJacobianLens
    from jlens_qwen.model_loader import load_qwen_coder

    lo, hi = [int(x) for x in args.layers.split("-")]
    layers = list(range(lo, hi + 1))
    bundle = load_qwen_coder(device=args.device, dtype=torch.float16)
    if args.lens.exists():
        lens = FittedJacobianLens.load(str(args.lens))
        use_j = True
    else:
        lens = identity_lens(bundle.d_model, layers=list(range(bundle.n_layers)))
        use_j = False

    result = lens.apply(
        bundle, args.prompt, layers=layers, max_seq_len=128, use_jacobian=use_j
    )
    tokens = [
        bundle.tokenizer.decode([int(t)]) for t in result["input_ids"][0].tolist()
    ]
    grid = {}
    for layer, logits in result["lens_logits"].items():
        row = []
        for t in range(logits.shape[0]):
            tid = int(logits[t].argmax())
            row.append(
                {
                    "token": bundle.tokenizer.decode([tid]),
                    "id": tid,
                    "score": float(logits[t].max()),
                }
            )
        grid[str(layer)] = row

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "prompt": args.prompt,
        "tokens": tokens,
        "use_jacobian": use_j,
        "grid": grid,
    }
    json_path = args.out_dir / "slice.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    html_lines = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>J-Lens Slice</title>",
        "<style>td{font-size:11px;padding:2px 4px;} .hl{background:#ffe08a;}</style>",
        "</head><body>",
        "<h1>J-Lens Layer × Position</h1>",
        f"<p>use_jacobian={use_j}</p>",
        f"<pre>{args.prompt}</pre>",
        "<table border='1'><tr><th>Layer</th>",
    ]
    for i, tok in enumerate(tokens):
        html_lines.append(f"<th title='pos {i}'>{tok!r}</th>")
    html_lines.append("</tr>")
    for layer in layers:
        html_lines.append(f"<tr><td class='hl'>{layer}</td>")
        for cell in grid[str(layer)]:
            html_lines.append(f"<td>{cell['token']}</td>")
        html_lines.append("</tr>")
    html_lines.append("</table></body></html>")
    html_path = args.out_dir / "slice.html"
    html_path.write_text("\n".join(html_lines), encoding="utf-8")
    print(f"Wrote {json_path} and {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
