#!/usr/bin/env python
"""Print indisputable proof summary for Aider+Qwen XPU harness."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    h = json.load(urllib.request.urlopen("http://127.0.0.1:8767/health", timeout=60))
    print("XPU_PROOF", h.get("device"), h.get("gpu_name"), "refuses_cpu", h.get("refuses_cpu_fallback"))

    cat = json.loads((ROOT / "experiments/dashboard/catalog.json").read_text(encoding="utf-8"))
    print("\nDASHBOARD_CATALOG")
    for r in cat:
        print(
            f"  {r['experiment']:32} kind={r['kind']:16} "
            f"inv={r['invocations']} layers={r['layers']} evidence={r['evidence']}"
        )

    print("\nHARD_AGENTIC_DETAIL")
    for name in [
        "aider_hard_dijkstra_grid",
        "aider_hard_token_bucket",
        "aider_hard_expr_compiler",
        "aider_hello_fibonacci",
        "aider_string_palindrome",
    ]:
        path = ROOT / f"experiments/out/{name}_latest.json"
        d = json.loads(path.read_text(encoding="utf-8"))
        invs = d.get("invocations") or []
        js_ok = sum(1 for i in invs if (i.get("jspace") or {}).get("layers"))
        lint = d.get("lint") or {}
        print(f"== {name} ==")
        print(
            f"  kind={d['kind']} device={d.get('device')} "
            f"n_inv={d['n_invocations']} jspace={js_ok}/{len(invs)}"
        )
        print(
            f"  layers_top={len(d.get('layers') or [])} lint_ok={lint.get('ok')} "
            f"evidence={d['evidence'].get('passed')}/{d['evidence'].get('total')}"
        )
        for inv in invs:
            layers = (inv.get("jspace") or {}).get("layers") or []
            layer_ids = [row.get("layer") for row in layers[:5]]
            print(
                f"  inv#{inv.get('invocation_index')}: "
                f"tok_out={inv.get('completion_tokens')} "
                f"jspace_layers={len(layers)} first_layers={layer_ids}"
            )


if __name__ == "__main__":
    main()
