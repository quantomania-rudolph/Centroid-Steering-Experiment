#!/usr/bin/env python
"""Human-readable J-lens vs logit-lens readout for one layer and one prompt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _decode_topk(bundle, logits_row, k: int = 5) -> list[dict]:
    import torch

    probs = torch.softmax(logits_row.float(), dim=-1)
    top = probs.topk(k)
    out = []
    for score, tid in zip(top.values.tolist(), top.indices.tolist(), strict=True):
        out.append(
            {
                "id": int(tid),
                "token": bundle.tokenizer.decode([int(tid)]),
                "prob": float(score),
            }
        )
    return out


def run_layer_readout(
    *,
    lens_path: Path,
    layer: int,
    prompt: str,
    max_seq_len: int = 128,
    top_k: int = 5,
    device: str = "cpu",
) -> dict:
    import torch

    from jlens_qwen.jacobian_lens import FittedJacobianLens
    from jlens_qwen.logit_lens import LogitLens
    from jlens_qwen.model_loader import load_qwen_coder

    lens = FittedJacobianLens.load(str(lens_path))
    if layer not in lens.jacobians:
        raise ValueError(f"layer {layer} not in lens; have {lens.source_layers}")

    bundle = load_qwen_coder(device=device, dtype=torch.float16, prefer_local=True)
    logit_lens = LogitLens(bundle)

    j_result = lens.apply(
        bundle,
        prompt,
        layers=[layer],
        max_seq_len=max_seq_len,
        use_jacobian=True,
    )
    ll_result = logit_lens.project_layers(
        prompt, layers=[layer], max_seq_len=max_seq_len
    )

    input_ids = j_result["input_ids"][0].tolist()
    tokens = [bundle.tokenizer.decode([int(t)]) for t in input_ids]
    n_pos = len(tokens)

    # Model logits aligned to positions (shift: logits[t] predicts token t+1)
    with torch.no_grad():
        model_out = bundle.model(
            input_ids=bundle.encode(prompt, max_length=max_seq_len),
            use_cache=False,
        )
        model_logits = model_out.logits[0].float().cpu()

    positions = []
    for pos in range(n_pos):
        ctx = "".join(tokens[: pos + 1])
        model_row = model_logits[pos] if pos < model_logits.shape[0] else None
        j_row = j_result["lens_logits"][layer][pos]
        ll_row = ll_result["lens_logits"][layer][pos]

        entry: dict = {
            "position": pos,
            "token": tokens[pos],
            "token_id": int(input_ids[pos]),
            "context": ctx,
            "j_lens_top": _decode_topk(bundle, j_row, top_k),
            "logit_lens_top": _decode_topk(bundle, ll_row, top_k),
        }
        if model_row is not None:
            entry["model_top"] = _decode_topk(bundle, model_row, top_k)
        positions.append(entry)

    return {
        "layer": layer,
        "lens_path": str(lens_path),
        "n_prompts_fitted": lens.n_prompts,
        "prompt": prompt,
        "max_seq_len": max_seq_len,
        "top_k": top_k,
        "positions": positions,
    }


def format_readout_text(report: dict) -> str:
    layer = report["layer"]
    lines = [
        f"Layer {layer} readout — J-lens vs logit lens vs model",
        f"Lens: {report['lens_path']} (fitted on {report['n_prompts_fitted']} prompt(s))",
        "",
        "Prompt:",
        report["prompt"],
        "",
        "=" * 72,
    ]
    for row in report["positions"]:
        lines.append(f"\nPosition {row['position']}: {row['token']!r}")
        lines.append(f"  Context: {row['context'][-80:]!r}")
        if "model_top" in row:
            m = row["model_top"][0]
            lines.append(
                f"  Model next-token:  {m['token']!r}  (p={m['prob']:.4f}, id={m['id']})"
            )
        ll = row["logit_lens_top"][0]
        j = row["j_lens_top"][0]
        lines.append(
            f"  Logit lens L{layer}: {ll['token']!r}  (p={ll['prob']:.4f}, id={ll['id']})"
        )
        lines.append(
            f"  J-lens L{layer}:    {j['token']!r}  (p={j['prob']:.4f}, id={j['id']})"
        )
        if row["j_lens_top"][0]["id"] != row["logit_lens_top"][0]["id"]:
            lines.append("  ^ J-lens differs from logit lens at this position")
        lines.append(f"  J-lens top-{report['top_k']}: " + ", ".join(
            f"{t['token']!r}" for t in row["j_lens_top"]
        ))
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show decoded token readouts for one layer on one prompt"
    )
    parser.add_argument(
        "--lens",
        type=Path,
        default=ROOT / "artifacts" / "lenses" / "qwen_jacobian_lens_L15.pt",
    )
    parser.add_argument("--layer", type=int, default=15)
    parser.add_argument("--prompts", type=Path, default=ROOT / "data" / "prompts" / "corpus.txt")
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--prompt", default=None, help="Override prompt text")
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "artifacts" / "readouts",
    )
    args = parser.parse_args()

    if not args.lens.exists():
        print(f"Lens not found: {args.lens}", file=sys.stderr)
        return 1

    if args.prompt is not None:
        prompt = args.prompt
    else:
        from jlens_qwen.jacobian_fit import load_prompts

        prompts = load_prompts(args.prompts)
        if not prompts:
            print("No prompts in corpus", file=sys.stderr)
            return 1
        if args.prompt_index < 0 or args.prompt_index >= len(prompts):
            print(f"prompt-index {args.prompt_index} out of range (0..{len(prompts)-1})")
            return 1
        prompt = prompts[args.prompt_index]

    report = run_layer_readout(
        lens_path=args.lens,
        layer=args.layer,
        prompt=prompt,
        max_seq_len=args.max_seq_len,
        top_k=args.top_k,
        device=args.device,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"L{args.layer}_readout"
    json_path = args.out_dir / f"{stem}.json"
    txt_path = args.out_dir / f"{stem}.txt"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    text = format_readout_text(report)
    txt_path.write_text(text, encoding="utf-8")
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))
    print(f"\nWrote {json_path}")
    print(f"Wrote {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
