#!/usr/bin/env python
"""Run Qwen on Intel Arc (XPU) as fast as practical, then J-lens L8-26 on the prompt."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_PROMPT = (ROOT / "experiments" / "prompts" / "explore_3d.txt").read_text(
    encoding="utf-8"
)
DEFAULT_LENS = ROOT / "artifacts" / "lenses" / "qwen_jacobian_lens.pt"
OUT_DIR = ROOT / "experiments" / "out"


def _configure_xpu_fast_path() -> dict:
    """Best-effort Arc speed knobs (safe no-ops if unsupported)."""
    import torch

    notes: list[str] = []
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
        notes.append("matmul_precision=high")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        # Prefer TF32-like / fast math paths when Intel exposes them
        try:
            torch.xpu.empty_cache()
            notes.append("xpu.empty_cache")
        except Exception:
            pass
    try:
        import intel_extension_for_pytorch as ipex  # type: ignore

        notes.append(f"ipex={getattr(ipex, '__version__', 'present')}")
    except ImportError:
        notes.append("ipex=not_installed")
    return {"notes": notes}


def _topk(bundle, logits_row, k: int = 5) -> list[dict]:
    import torch

    probs = torch.softmax(logits_row.float(), dim=-1)
    top = probs.topk(k)
    return [
        {
            "id": int(tid),
            "token": bundle.tokenizer.decode([int(tid)]),
            "prob": float(score),
        }
        for score, tid in zip(top.values.tolist(), top.indices.tolist(), strict=True)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="xpu", help="xpu | cpu | auto")
    parser.add_argument("--lens", type=Path, default=DEFAULT_LENS)
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument(
        "--experiment",
        default=None,
        help="Output slug (default: prompt file stem, else explore_3d)",
    )
    parser.add_argument("--layers", default="8-26")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Generation length cap (default 4096; raise carefully vs Arc VRAM)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=8192,
        help="Max prompt+context tokens for encode/J-lens (default 8192)",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Only run J-lens on the prompt (no completion)",
    )
    args = parser.parse_args()

    import torch

    from jlens_qwen.jacobian_lens import FittedJacobianLens
    from jlens_qwen.logit_lens import LogitLens
    from jlens_qwen.model_loader import load_qwen_coder

    prompt = (
        args.prompt_file.read_text(encoding="utf-8").strip()
        if args.prompt_file
        else DEFAULT_PROMPT.strip()
    )
    if args.experiment:
        experiment = args.experiment
    elif args.prompt_file:
        experiment = args.prompt_file.stem
    else:
        experiment = "explore_3d"
    lo, hi = [int(x) for x in args.layers.split("-")]
    layers = list(range(lo, hi + 1))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    fast = _configure_xpu_fast_path()
    print("Fast-path notes:", "; ".join(fast["notes"]), flush=True)

    if not args.lens.is_file():
        print(f"Missing lens: {args.lens}", file=sys.stderr)
        return 1

    t_load0 = time.perf_counter()
    bundle = load_qwen_coder(
        device=args.device, dtype=torch.float16, prefer_local=True
    )
    # Optional IPEX optimize (inference)
    try:
        import intel_extension_for_pytorch as ipex  # type: ignore

        bundle.model = ipex.optimize(bundle.model, dtype=torch.float16, inplace=True)
        print("Applied ipex.optimize(fp16)", flush=True)
    except Exception as exc:
        print(f"IPEX optimize skipped: {exc}", flush=True)

    load_s = time.perf_counter() - t_load0
    print(
        f"Loaded {bundle.model_id} on {bundle.device} dtype={bundle.dtype} in {load_s:.1f}s",
        flush=True,
    )
    if str(bundle.device) != "xpu":
        print(
            "WARNING: not on XPU — Arc may be unavailable; continuing on "
            f"{bundle.device}",
            flush=True,
        )

    lens = FittedJacobianLens.load(str(args.lens))
    missing = [L for L in layers if L not in lens.jacobians]
    if missing:
        print(f"Lens missing layers {missing}; have {lens.source_layers}", file=sys.stderr)
        return 1

    # Chat-style prompt for Instruct model
    messages = [
        {
            "role": "system",
            "content": "You are a careful coding assistant. Write correct, concise Python.",
        },
        {"role": "user", "content": prompt},
    ]
    chat_text = bundle.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    generation = {"skipped": True}
    if not args.skip_generate:
        input_ids = bundle.tokenizer(
            chat_text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_seq_len,
        ).input_ids.to(bundle.device)

        # Warmup generate (1 token) — reduces first-token latency on XPU
        with torch.inference_mode():
            _ = bundle.model.generate(
                input_ids=input_ids[:, : min(16, input_ids.shape[1])],
                max_new_tokens=1,
                do_sample=False,
                use_cache=True,
            )
        if str(bundle.device).startswith("xpu"):
            torch.xpu.synchronize()

        t0 = time.perf_counter()
        with torch.inference_mode():
            out_ids = bundle.model.generate(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=bundle.tokenizer.eos_token_id,
            )
        if str(bundle.device).startswith("xpu"):
            torch.xpu.synchronize()
        gen_s = time.perf_counter() - t0
        new_tokens = out_ids[0, input_ids.shape[1] :]
        completion = bundle.tokenizer.decode(new_tokens, skip_special_tokens=True)
        toks = int(new_tokens.numel())
        generation = {
            "skipped": False,
            "seconds": gen_s,
            "new_tokens": toks,
            "tokens_per_sec": (toks / gen_s) if gen_s > 0 else None,
            "completion": completion,
        }
        print(
            f"Generate: {toks} tokens in {gen_s:.2f}s "
            f"({generation['tokens_per_sec']:.2f} tok/s)"
            if generation["tokens_per_sec"]
            else f"Generate: {toks} tokens in {gen_s:.2f}s",
            flush=True,
        )
        print("\n===== MODEL COMPLETION =====\n", flush=True)
        print(completion, flush=True)
        print("\n===== END COMPLETION =====\n", flush=True)

    # J-lens: one forward for all layers on the user coding prompt (chat-formatted)
    print(f"J-lens apply layers {layers[0]}-{layers[-1]} ...", flush=True)
    t1 = time.perf_counter()
    with torch.inference_mode():
        j_result = lens.apply(
            bundle,
            chat_text,
            layers=layers,
            positions=[-1],
            max_seq_len=args.max_seq_len,
            use_jacobian=True,
        )
        logit = LogitLens(bundle)
        ll_result = logit.project_layers(
            chat_text, layers=layers, positions=[-1], max_seq_len=args.max_seq_len
        )
    if str(bundle.device).startswith("xpu"):
        torch.xpu.synchronize()
    jlens_s = time.perf_counter() - t1

    layer_rows = []
    for layer in layers:
        j_row = j_result["lens_logits"][layer][0]
        ll_row = ll_result["lens_logits"][layer][0]
        j_top = _topk(bundle, j_row, args.top_k)
        ll_top = _topk(bundle, ll_row, args.top_k)
        layer_rows.append(
            {
                "layer": layer,
                "j_lens_top": j_top,
                "logit_lens_top": ll_top,
                "j_differs_from_logit": j_top[0]["id"] != ll_top[0]["id"],
            }
        )

    # Human text summary
    lines = [
        f"Experiment: {experiment} @ {stamp}",
        f"Device: {bundle.device}  dtype: {bundle.dtype}  load_s: {load_s:.1f}",
        f"Lens: {args.lens}  n_prompts_fitted={lens.n_prompts}",
        f"J-lens forward (all layers): {jlens_s:.2f}s",
        "",
        "----- PROMPT -----",
        prompt,
        "",
    ]
    if not generation.get("skipped"):
        lines += [
            "----- COMPLETION -----",
            generation["completion"],
            "",
            f"gen_tokens={generation['new_tokens']}  "
            f"gen_s={generation['seconds']:.2f}  "
            f"tok/s={generation.get('tokens_per_sec')}",
            "",
        ]
    lines += [
        "----- J-LENS vs LOGIT-LENS (last prompt token) -----",
        f"{'L':>3}  {'J-lens top1':<22} {'p':>6}  {'logit-lens top1':<22} {'p':>6}  diff",
    ]
    for row in layer_rows:
        jt = row["j_lens_top"][0]
        lt = row["logit_lens_top"][0]
        jtok = repr(jt["token"])[:20]
        ltok = repr(lt["token"])[:20]
        lines.append(
            f"{row['layer']:3d}  {jtok:<22} {jt['prob']:6.3f}  "
            f"{ltok:<22} {lt['prob']:6.3f}  "
            f"{'Y' if row['j_differs_from_logit'] else 'n'}"
        )
        # top-5 compact
        j5 = ", ".join(repr(t["token"]) for t in row["j_lens_top"])
        lines.append(f"     J top5: {j5}")

    report_txt = "\n".join(lines) + "\n"
    txt_path = args.out_dir / f"{experiment}_{stamp}.txt"
    json_path = args.out_dir / f"{experiment}_{stamp}.json"
    latest_txt = args.out_dir / f"{experiment}_latest.txt"
    latest_json = args.out_dir / f"{experiment}_latest.json"

    payload = {
        "experiment": experiment,
        "stamp": stamp,
        "device": str(bundle.device),
        "dtype": str(bundle.dtype),
        "load_seconds": load_s,
        "fast_path": fast,
        "prompt": prompt,
        "chat_text_preview": chat_text[:500],
        "generation": generation,
        "jlens_seconds": jlens_s,
        "layers": layer_rows,
        "lens_path": str(args.lens),
        "n_prompts_fitted": lens.n_prompts,
    }
    txt_path.write_text(report_txt, encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest_txt.write_text(report_txt, encoding="utf-8")
    latest_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Windows consoles may be cp1252; never crash on exotic token glyphs.
    try:
        print(report_txt)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(report_txt.encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")
    print(f"Wrote {txt_path}")
    print(f"Wrote {json_path}")

    dash_script = ROOT / "experiments" / "build_dashboard.py"
    if dash_script.is_file():
        import subprocess

        rc = subprocess.call([sys.executable, str(dash_script)], cwd=str(ROOT))
        if rc == 0:
            print(f"Dashboard: {ROOT / 'experiments' / 'dashboard' / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
