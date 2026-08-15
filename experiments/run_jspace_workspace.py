#!/usr/bin/env python
"""Full J-space workspace experiment: generation + per-layer top-20 readouts + evidence."""

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
JSPACE_TOP_K = 20


def _safe_print(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(msg.encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")


def _configure_fast_path() -> list[str]:
    import torch

    notes: list[str] = []
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
        notes.append("matmul_precision=high")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
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
    return notes


def _topk_decode(bundle, logits_row, k: int) -> list[dict]:
    import torch

    probs = torch.softmax(logits_row.float(), dim=-1)
    top = probs.topk(min(k, probs.shape[-1]))
    return [
        {
            "id": int(tid),
            "token": bundle.tokenizer.decode([int(tid)]),
            "prob": float(score),
        }
        for score, tid in zip(top.values.tolist(), top.indices.tolist(), strict=True)
    ]


def _mean_topk(bundle, logits_matrix, k: int) -> list[dict]:
    import torch

    probs = torch.softmax(logits_matrix.float(), dim=-1).mean(dim=0)
    top = probs.topk(min(k, probs.shape[-1]))
    return [
        {
            "id": int(tid),
            "token": bundle.tokenizer.decode([int(tid)]),
            "prob": float(score),
        }
        for score, tid in zip(top.values.tolist(), top.indices.tolist(), strict=True)
    ]


def _sparse_at_position(bundle, h, lens, layer: int, k: int = 20) -> dict | None:
    from jlens_qwen.jspace_decompose import decompose_activation

    if layer not in lens.jacobians:
        return None
    decomp = decompose_activation(
        h.cpu(),
        bundle.W_U.cpu(),
        lens.jacobians[layer].cpu(),
        k=k,
        max_dict=2048,
    )
    tokens = []
    for vid, coef in zip(
        decomp["vocab_indices"].tolist(),
        decomp["coefficients"].tolist(),
        strict=True,
    ):
        tokens.append(
            {
                "id": int(vid),
                "token": bundle.tokenizer.decode([int(vid)]),
                "coeff": float(coef),
            }
        )
    return {
        "relative_mse": float(decomp["relative_mse"]),
        "atoms": tokens,
    }


def _build_evidence(
    bundle,
    layer_rows: list[dict],
    workspace: dict,
    generation: dict,
    prompt_len: int,
    model_logits_at_prompt_end: object,
) -> dict:
    import torch

    evidence: dict = {"checks": []}

    # E1: workspace band detected
    band = workspace.get("workspace_band")
    evidence["checks"].append(
        {
            "id": "E1",
            "name": "workspace band detected",
            "pass": band is not None and band[1] > band[0],
            "detail": f"band={band} onset={workspace.get('onset_layer')} motor={workspace.get('motor_layer')}",
        }
    )

    # E2: late-layer topk_acc high (motor layer)
    motor = workspace.get("motor_layer")
    if motor is not None:
        motor_metrics = workspace.get("per_layer", {}).get(str(motor), {})
        late_acc = motor_metrics.get("topk_acc", 0.0)
        evidence["checks"].append(
            {
                "id": "E2",
                "name": f"L{motor} topk_acc >= 0.5",
                "pass": late_acc >= 0.5,
                "detail": f"topk_acc={late_acc:.4f}",
            }
        )

    # E3: J-lens at prompt-end matches model next-token in top-5
    if model_logits_at_prompt_end is not None:
        model_top5 = _topk_decode(bundle, model_logits_at_prompt_end, 5)
        model_top1_id = model_top5[0]["id"]
        # find layer 24 or highest workspace layer in rows
        late_row = next(
            (r for r in reversed(layer_rows) if r["layer"] >= 22),
            layer_rows[-1] if layer_rows else None,
        )
        if late_row:
            j_top5_ids = {t["id"] for t in late_row.get("prompt_end_top20", [])[:5]}
            hit = model_top1_id in j_top5_ids
            evidence["checks"].append(
                {
                    "id": "E3",
                    "name": f"L{late_row['layer']} J-lens top-5 contains model next-token",
                    "pass": hit,
                    "detail": f"model_top1={model_top5[0]['token']!r} j_top5_ids={list(j_top5_ids)[:5]}",
                }
            )

    # E4: mid-layer J differs from logit-lens (divergence)
    mid_rows = [r for r in layer_rows if 14 <= r["layer"] <= 18]
    diverged = 0
    for r in mid_rows:
        jt = r.get("prompt_end_top20", [{}])[0].get("id")
        lt = r.get("prompt_end_logit_top20", [{}])[0].get("id")
        if jt is not None and lt is not None and jt != lt:
            diverged += 1
    evidence["checks"].append(
        {
            "id": "E4",
            "name": "mid-layer J-lens != logit-lens at prompt end",
            "pass": diverged >= 2,
            "detail": f"diverged_layers={diverged}/{len(mid_rows)}",
        }
    )

    # E5: sparse reconstruction at workspace layer
    sparse_ok = False
    for r in layer_rows:
        sp = r.get("sparse_prompt_end")
        if sp and sp.get("relative_mse", 1.0) < 0.95:
            sparse_ok = True
            break
    evidence["checks"].append(
        {
            "id": "E5",
            "name": "sparse J-space reconstruction rel_mse < 0.95",
            "pass": sparse_ok,
            "detail": "checked workspace layers at prompt end",
        }
    )

    # E6: generation contains code markers
    completion = generation.get("completion", "") if not generation.get("skipped") else ""
    code_markers = ["def ", "import ", "class ", "return ", "```"]
    found_markers = [m for m in code_markers if m in completion]
    evidence["checks"].append(
        {
            "id": "E6",
            "name": "generation contains Python code markers",
            "pass": len(found_markers) >= 2,
            "detail": f"markers={found_markers} tokens={generation.get('new_tokens')}",
        }
    )

    # E7: trajectory shows convergence in workspace band
    conv_layers = 0
    for r in layer_rows:
        traj = r.get("trajectory_top1", [])
        if len(traj) < 3:
            continue
        # check if generation region (after prompt) has stable/readable tokens
        gen_traj = traj[prompt_len:prompt_len + 20]
        if not gen_traj:
            continue
        readable = sum(
            1
            for t in gen_traj
            if t and len(t.strip()) > 0 and not all(ord(c) > 127 for c in t.strip())
        )
        if readable >= len(gen_traj) * 0.6:
            conv_layers += 1
    evidence["checks"].append(
        {
            "id": "E7",
            "name": "workspace layers have readable generation trajectory",
            "pass": conv_layers >= 5,
            "detail": f"readable_layers={conv_layers}",
        }
    )

    evidence["overall_pass"] = all(c["pass"] for c in evidence["checks"])
    evidence["passed"] = sum(1 for c in evidence["checks"] if c["pass"])
    evidence["total"] = len(evidence["checks"])
    return evidence


def run_experiment(args: argparse.Namespace) -> int:
    import torch

    from jlens_qwen.jacobian_lens import FittedJacobianLens
    from jlens_qwen.logit_lens import LogitLens
    from jlens_qwen.model_loader import ActivationRecorder, load_qwen_coder
    from jlens_qwen.workspace_detect import compute_layer_metrics

    prompt = (
        args.prompt_file.read_text(encoding="utf-8").strip()
        if args.prompt_file
        else DEFAULT_PROMPT.strip()
    )
    experiment = args.experiment or (
        args.prompt_file.stem if args.prompt_file else "new_full_3d"
    )
    lo, hi = [int(x) for x in args.layers.split("-")]
    layers = list(range(lo, hi + 1))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    fast_notes = _configure_fast_path()
    print("Fast-path:", "; ".join(fast_notes), flush=True)

    if not args.lens.is_file():
        print(f"Missing lens: {args.lens}", file=sys.stderr)
        return 1

    t_load0 = time.perf_counter()
    bundle = load_qwen_coder(
        device=args.device, dtype=torch.float16, prefer_local=True
    )
    try:
        import intel_extension_for_pytorch as ipex  # type: ignore

        bundle.model = ipex.optimize(bundle.model, dtype=torch.float16, inplace=True)
        print("Applied ipex.optimize(fp16)", flush=True)
    except Exception as exc:
        print(f"IPEX optimize skipped: {exc}", flush=True)

    load_s = time.perf_counter() - t_load0
    print(
        f"Loaded on {bundle.device} dtype={bundle.dtype} in {load_s:.1f}s",
        flush=True,
    )

    lens = FittedJacobianLens.load(str(args.lens))
    missing = [L for L in layers if L not in lens.jacobians]
    if missing:
        print(f"Lens missing layers {missing}", file=sys.stderr)
        return 1

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

    # --- Generation ---
    generation: dict = {"skipped": True}
    full_text = chat_text
    prompt_len_tokens = 0

    if not args.skip_generate:
        input_ids = bundle.tokenizer(
            chat_text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_seq_len,
        ).input_ids.to(bundle.device)
        prompt_len_tokens = int(input_ids.shape[1])

        # warmup
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
        full_text = chat_text + completion
        print(
            f"Generated {toks} tokens in {gen_s:.2f}s "
            f"({generation['tokens_per_sec']:.2f} tok/s)",
            flush=True,
        )

    # --- J-space readouts on full sequence (truncate for VRAM) ---
    readout_text = full_text
    readout_ids = bundle.tokenizer(
        readout_text, return_tensors="pt", truncation=True, max_length=args.max_jspace_seq_len
    ).input_ids[0]
    readout_len = int(readout_ids.shape[0])
    readout_text = bundle.tokenizer.decode(readout_ids.tolist())
    print(
        f"J-space readout length: {readout_len} tokens "
        f"(max_jspace_seq_len={args.max_jspace_seq_len})",
        flush=True,
    )

    print(f"J-space readout layers {layers[0]}-{layers[-1]} ...", flush=True)
    t1 = time.perf_counter()
    logit_lens = LogitLens(bundle)

    if str(bundle.device).startswith("xpu"):
        torch.xpu.empty_cache()

    with torch.inference_mode():
        j_result = lens.apply(
            bundle,
            readout_text,
            layers=layers,
            max_seq_len=args.max_jspace_seq_len,
            use_jacobian=True,
        )
    if str(bundle.device).startswith("xpu"):
        torch.xpu.empty_cache()

    # Logit-lens only on prompt for comparison (saves VRAM vs full sequence)
    with torch.inference_mode():
        ll_result = logit_lens.project_layers(
            chat_text,
            layers=layers,
            max_seq_len=min(args.max_jspace_seq_len, 2048),
        )
    if str(bundle.device).startswith("xpu"):
        torch.xpu.synchronize()
    jlens_s = time.perf_counter() - t1

    input_ids = j_result["input_ids"][0]
    token_strs = [bundle.tokenizer.decode([int(t)]) for t in input_ids.tolist()]
    seq_len = len(token_strs)
    if prompt_len_tokens == 0:
        # estimate prompt length from chat_text only encode
        prompt_len_tokens = int(
            bundle.tokenizer(chat_text, return_tensors="pt").input_ids.shape[1]
        )

    prompt_end_pos = max(0, min(prompt_len_tokens - 1, seq_len - 1))
    gen_start_pos = min(prompt_len_tokens, seq_len - 1)
    gen_mid_pos = min(
        prompt_len_tokens + max(1, generation.get("new_tokens", 0) // 2),
        seq_len - 1,
    )
    ll_prompt_end = max(0, min(prompt_len_tokens - 1, ll_result["lens_logits"][layers[0]].shape[0] - 1))

    # Activations for sparse decompose (workspace layers only)
    workspace_layers_for_sparse = [20, 22, 24]
    sparse_layers = [l for l in workspace_layers_for_sparse if l in layers]
    activations: dict[int, torch.Tensor] = {}
    if sparse_layers:
        with ActivationRecorder(bundle.layers, at=sparse_layers) as rec:
            ids_enc = bundle.encode(readout_text, max_length=args.max_jspace_seq_len)
            with torch.inference_mode():
                _ = bundle.model(input_ids=ids_enc, use_cache=False)
            if str(bundle.device).startswith("xpu"):
                torch.xpu.empty_cache()
        for l in sparse_layers:
            activations[l] = rec.activations[l][0].cpu()

    model_logits_at_prompt_end = None
    if j_result.get("model_logits") is not None:
        ml = j_result["model_logits"]
        if prompt_end_pos < ml.shape[0]:
            model_logits_at_prompt_end = ml[prompt_end_pos]

    layer_rows: list[dict] = []
    stride = max(1, seq_len // args.max_position_samples)

    for layer in layers:
        j_logits = j_result["lens_logits"][layer]
        ll_logits = ll_result["lens_logits"][layer]
        T = j_logits.shape[0]

        trajectory = []
        for t in range(T):
            top1 = _topk_decode(bundle, j_logits[t], 1)
            trajectory.append(top1[0]["token"] if top1 else "")

        row: dict = {
            "layer": layer,
            "seq_len": T,
            "trajectory_top1": trajectory,
            "mean_top20": _mean_topk(bundle, j_logits, JSPACE_TOP_K),
            "prompt_end_top20": _topk_decode(bundle, j_logits[prompt_end_pos], JSPACE_TOP_K),
            "prompt_end_logit_top20": _topk_decode(
                bundle, ll_logits[ll_prompt_end], JSPACE_TOP_K
            ),
            "gen_start_top20": _topk_decode(bundle, j_logits[gen_start_pos], JSPACE_TOP_K),
            "gen_mid_top20": _topk_decode(bundle, j_logits[gen_mid_pos], JSPACE_TOP_K),
        }

        if layer in activations:
            h_pe = activations[layer][prompt_end_pos]
            row["sparse_prompt_end"] = _sparse_at_position(
                bundle, h_pe, lens, layer, k=JSPACE_TOP_K
            )
            h_gs = activations[layer][gen_start_pos]
            row["sparse_gen_start"] = _sparse_at_position(
                bundle, h_gs, lens, layer, k=JSPACE_TOP_K
            )

        # sampled positions for trajectory detail
        sampled: list[dict] = []
        for t in range(0, T, stride):
            sampled.append(
                {
                    "pos": t,
                    "token": token_strs[t],
                    "j_top20": _topk_decode(bundle, j_logits[t], JSPACE_TOP_K),
                }
            )
        row["sampled_positions"] = sampled
        layer_rows.append(row)
        _safe_print(f"  L{layer}: traj_len={T} mean_top1={row['mean_top20'][0]['token']!r}")

    # Workspace metrics on this prompt
    print("Computing workspace metrics ...", flush=True)
    workspace = compute_layer_metrics(
        bundle,
        lens,
        [readout_text],
        layers=list(range(0, bundle.n_layers, 2)),
        max_seq_len=min(args.max_jspace_seq_len, 512),
        use_jacobian=True,
    )

    evidence = _build_evidence(
        bundle,
        layer_rows,
        workspace,
        generation,
        prompt_len_tokens,
        model_logits_at_prompt_end,
    )

    payload = {
        "experiment": experiment,
        "stamp": stamp,
        "kind": "jspace_workspace",
        "device": str(bundle.device),
        "dtype": str(bundle.dtype),
        "load_seconds": load_s,
        "fast_path": {"notes": fast_notes},
        "prompt": prompt,
        "prompt_len_tokens": prompt_len_tokens,
        "seq_len": seq_len,
        "token_preview": token_strs[:30],
        "generation": generation,
        "jlens_seconds": jlens_s,
        "jspace_top_k": JSPACE_TOP_K,
        "layers": layer_rows,
        "workspace": workspace,
        "evidence": evidence,
        "lens_path": str(args.lens),
        "n_prompts_fitted": lens.n_prompts,
        "key_positions": {
            "prompt_end": prompt_end_pos,
            "gen_start": gen_start_pos,
            "gen_mid": gen_mid_pos,
        },
    }

    json_path = args.out_dir / f"{experiment}_{stamp}.json"
    txt_path = args.out_dir / f"{experiment}_{stamp}.txt"
    latest_json = args.out_dir / f"{experiment}_latest.json"

    # Text summary
    lines = [
        f"J-Space Workspace Experiment: {experiment} @ {stamp}",
        f"Device: {bundle.device}  seq_len={seq_len}  prompt_len={prompt_len_tokens}",
        f"Workspace band: {workspace.get('workspace_band')}  "
        f"onset={workspace.get('onset_layer')} motor={workspace.get('motor_layer')}",
        f"Evidence: {evidence['passed']}/{evidence['total']} checks pass",
        "",
    ]
    for c in evidence["checks"]:
        mark = "PASS" if c["pass"] else "FAIL"
        lines.append(f"  [{mark}] {c['id']}: {c['name']} — {c['detail']}")
    lines.append("")
    lines.append("--- Mean top-20 J-space representations per layer ---")
    for row in layer_rows:
        top3 = ", ".join(repr(t["token"]) for t in row["mean_top20"][:3])
        lines.append(f"  L{row['layer']:2d}: {top3}")
    report_txt = "\n".join(lines) + "\n"

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    txt_path.write_text(report_txt, encoding="utf-8")
    latest_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    _safe_print(report_txt)

    print(f"Wrote {json_path}")
    print(f"Evidence: {evidence['passed']}/{evidence['total']} PASS")

    dash_script = ROOT / "experiments" / "build_dashboard.py"
    if dash_script.is_file():
        import subprocess

        subprocess.call([sys.executable, str(dash_script)], cwd=str(ROOT))
    return 0 if evidence["overall_pass"] else 0  # still return 0; evidence in JSON


def main() -> int:
    parser = argparse.ArgumentParser(description="Full J-space workspace experiment")
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--lens", type=Path, default=DEFAULT_LENS)
    parser.add_argument("--prompt-file", type=Path, default=None)
    parser.add_argument("--experiment", default="new_full_3d")
    parser.add_argument("--layers", default="8-26")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--max-jspace-seq-len", type=int, default=2048)
    parser.add_argument("--max-position-samples", type=int, default=40)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--skip-generate", action="store_true")
    args = parser.parse_args()
    return run_experiment(args)


if __name__ == "__main__":
    raise SystemExit(main())
