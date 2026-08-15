"""Natural pair miner — tracer corpus scan, validate, capture, admit."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from jlens_qwen.caa.admission import admission_checks, verify_capture_integrity
from jlens_qwen.caa.activation_capture import capture_matched_pair_activations
from jlens_qwen.caa.calm_tail import build_calm_tail_from_file
from jlens_qwen.caa.calm_tail_registry import find_calm_tail
from jlens_qwen.caa.capture_manifest import CaptureManifest, write_admission_gate, write_manifest
from jlens_qwen.caa.kv_stamp import read_global_kv_live_gate
from jlens_qwen.caa.method_b_databus import build_method_b_record
from jlens_qwen.caa.miner_pairing import (
    build_clean_chaos_from_token_ids,
    check_d_coherence,
    dense_token_ids_ok,
    extract_d_from_lcp,
    make_pair_slug,
    resolve_prompt_ids_from_session,
)
from jlens_qwen.chaos.constants import PROBE_LAYERS

Mode = Literal["scan", "dry_run", "capture"]


def _pool_eligible_on_disk(out_dir: Path) -> bool:
    """True only when an existing capture is already admitted into the natural pool."""
    act_ok = (out_dir / "activations_at_d.pt").is_file() or (out_dir / "activations_at_d.json").is_file()
    if not act_ok:
        return False
    adm = out_dir / "admission_gate.json"
    if not adm.is_file():
        return False
    try:
        data = json.loads(adm.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return bool(data.get("pass") and data.get("pool_eligible") and data.get("source") == "natural")


def _purge_capture_dir(out_dir: Path) -> None:
    """Remove a partial/failed capture so skip-if-exists cannot freeze a bad slug."""
    if out_dir.is_dir():
        shutil.rmtree(out_dir, ignore_errors=True)


@dataclass
class MinerRejection:
    experiment: str
    invocation_id: str
    task_slug: str
    reason: str
    detail: str = ""


@dataclass
class MinerCandidate:
    experiment: str
    invocation_id: str
    task_slug: str
    d_tracer: int
    prefix_len: int
    confidence: str
    token_ids_match: bool
    calm_tail_path: str | None = None


@dataclass
class MinerRunResult:
    mode: Mode
    n_scanned: int
    n_eligible: int
    n_candidates: int
    n_admitted: int
    n_rejected: int
    candidates: list[MinerCandidate] = field(default_factory=list)
    rejections: list[MinerRejection] = field(default_factory=list)
    admitted_slugs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "n_scanned": self.n_scanned,
            "n_eligible": self.n_eligible,
            "n_candidates": self.n_candidates,
            "n_admitted": self.n_admitted,
            "n_rejected": self.n_rejected,
            "candidates": [c.__dict__ for c in self.candidates],
            "rejections": [
                {
                    "experiment": r.experiment,
                    "invocation_id": r.invocation_id,
                    "task_slug": r.task_slug,
                    "reason": r.reason,
                    "detail": r.detail,
                }
                for r in self.rejections
            ],
            "admitted_slugs": self.admitted_slugs,
        }


def _load_experiment_files(corpus_glob: str, root: Path) -> list[Path]:
    return sorted(root.glob(corpus_glob))


def scan_corpus(
    *,
    root: Path,
    corpus_glob: str = "experiments/out/*_latest.json",
    trace_invocation_fn: Any,
    calm_tails_root: Path,
) -> MinerRunResult:
    """Tracer-only scan: list eligible panic invocations and calm-tail availability."""
    result = MinerRunResult(mode="scan", n_scanned=0, n_eligible=0, n_candidates=0, n_admitted=0, n_rejected=0)
    for path in _load_experiment_files(corpus_glob, root):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        experiment = str(data.get("experiment") or path.stem)
        task_slug = str(data.get("task_slug") or "")
        for inv in data.get("invocations") or []:
            result.n_scanned += 1
            inv_id = str(inv.get("id") or "")
            trace = trace_invocation_fn(
                inv,
                experiment=experiment,
                task_slug=task_slug,
                experiment_data=data,
                use_tokenizer=False,
            )
            if not trace.eligible or trace.d_consensus is None:
                continue
            result.n_eligible += 1
            if not dense_token_ids_ok(trace):
                result.rejections.append(
                    MinerRejection(
                        experiment,
                        inv_id,
                        task_slug,
                        "sparse_token_ids",
                        f"token_ids_match={trace.token_ids_match}",
                    )
                )
                result.n_rejected += 1
                continue
            calm_path = find_calm_tail(task_slug, inv_id, int(trace.d_consensus), root=calm_tails_root)
            cand = MinerCandidate(
                experiment=experiment,
                invocation_id=inv_id,
                task_slug=task_slug,
                d_tracer=int(trace.d_consensus),
                prefix_len=len(trace.prefix_token_ids or []),
                confidence=str(trace.confidence),
                token_ids_match=bool(trace.token_ids_match),
                calm_tail_path=str(calm_path) if calm_path else None,
            )
            result.candidates.append(cand)
            result.n_candidates += 1
            if calm_path is None:
                result.rejections.append(
                    MinerRejection(
                        experiment,
                        inv_id,
                        task_slug,
                        "missing_calm_tail",
                        f"expected under {calm_tails_root}",
                    )
                )
                result.n_rejected += 1
    return result


def mine_corpus(
    *,
    root: Path,
    natural_root: Path,
    corpus_glob: str,
    trace_invocation_fn: Any,
    validate_pair_fn: Any,
    resolve_assistant_text_fn: Any,
    calm_tails_root: Path,
    bundle: Any | None = None,
    mode: Mode = "dry_run",
    tokenizer: Any | None = None,
    max_pairs: int | None = None,
    only_tasks: set[str] | None = None,
) -> MinerRunResult:
    """Dry-run validates pairs; capture mode writes pairs/natural/ when bundle provided.

    Dry-run n_admitted = count of pair-validated slugs (not on-disk pool captures).
    Capture n_admitted = pool-eligible activations written under natural_root.
    """
    scan = scan_corpus(
        root=root,
        corpus_glob=corpus_glob,
        trace_invocation_fn=trace_invocation_fn,
        calm_tails_root=calm_tails_root,
    )
    scan.mode = mode
    if mode == "scan":
        return scan

    kv_info = read_global_kv_live_gate(root / "artifacts" / "gates" / "kv_equivalence_live_gate.json")
    kv_ok = bool(kv_info.get("pass"))
    if mode == "capture" and not kv_ok:
        scan.rejections.append(
            MinerRejection("", "", "", "kv_live_gate", "kv_equivalence_live_gate.json not pass")
        )
        return scan

    admitted = 0
    for cand in list(scan.candidates):
        if only_tasks is not None and cand.task_slug not in only_tasks:
            continue
        if max_pairs is not None and admitted >= max_pairs:
            break
        if not cand.calm_tail_path:
            continue
        exp_data: dict[str, Any] | None = None
        inv: dict[str, Any] | None = None
        for p in _load_experiment_files(corpus_glob, root):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if str(data.get("experiment") or "") != cand.experiment:
                continue
            for row in data.get("invocations") or []:
                if str(row.get("id") or "") == cand.invocation_id:
                    exp_data = data
                    inv = row
                    break
            if inv is not None:
                break
        if inv is None or exp_data is None:
            scan.rejections.append(
                MinerRejection(cand.experiment, cand.invocation_id, cand.task_slug, "invocation_not_found", "")
            )
            scan.n_rejected += 1
            continue

        trace = trace_invocation_fn(
            inv,
            experiment=cand.experiment,
            task_slug=cand.task_slug,
            experiment_data=exp_data,
            use_tokenizer=False,
        )
        bad_ids = list((inv.get("chaos") or {}).get("token_ids") or [])
        prefix_ids = list(trace.prefix_token_ids or bad_ids[: int(trace.d_consensus or 0)])
        d_tracer = int(trace.d_consensus or cand.d_tracer)

        try:
            calm_spec = build_calm_tail_from_file(
                prefix_token_ids=prefix_ids,
                d=d_tracer,
                calm_completion_path=cand.calm_tail_path,
                tokenizer=tokenizer,
            )
        except (ValueError, FileNotFoundError) as exc:
            scan.rejections.append(
                MinerRejection(cand.experiment, cand.invocation_id, cand.task_slug, "calm_tail_invalid", str(exc))
            )
            scan.n_rejected += 1
            continue

        d_extract = extract_d_from_lcp(calm_spec.calm_full_token_ids, bad_ids)
        d_chk = check_d_coherence(d_extract, d_tracer)
        if not d_chk.passed:
            scan.rejections.append(
                MinerRejection(cand.experiment, cand.invocation_id, cand.task_slug, "d_coherence", d_chk.detail)
            )
            scan.n_rejected += 1
            continue

        # Prefer LCP extract d for all pair fields; calm_spec was built at tracer d,
        # so rebuild prefix/tails from full ids when extract differs.
        d_use = d_extract
        calm_full = list(calm_spec.calm_full_token_ids)
        prefix_use = calm_full[:d_use]
        calm_tail_ids = calm_full[d_use:]
        bad_tail_ids = bad_ids[d_use:]

        def _decode_ids(ids: list[int]) -> str:
            if tokenizer is None or not ids:
                return ""
            try:
                return str(tokenizer.decode(ids, skip_special_tokens=True))
            except TypeError:
                return str(tokenizer.decode(ids))

        prompt_len = int(inv.get("prompt_tokens") or inv.get("prompt_len") or 0)
        bad_text = resolve_assistant_text_fn(inv, exp_data)
        calm_tail_text = _decode_ids(calm_tail_ids) or calm_spec.calm_text
        # Decode-only: never character-slice with a token index (fail closed into validate).
        bad_tail_text = _decode_ids(bad_tail_ids)
        spec = {
            "d": d_use,
            "prefix_token_ids": prefix_use,
            "calm_tail_token_ids": calm_tail_ids,
            "bad_tail_token_ids": bad_tail_ids,
            "calm_tail_text": calm_tail_text,
            "bad_tail_text": bad_tail_text,
            "d_tracer": d_tracer,
            "d_extract": d_extract,
        }
        # Force method_b records to use extract d + remapped prefix (not tracer).
        divergence_for_mb = dict(trace.to_dict())
        divergence_for_mb["d_consensus"] = d_use
        divergence_for_mb["prefix_token_ids"] = prefix_use
        calm_chaos = build_clean_chaos_from_token_ids(calm_full)
        calm_mb = build_method_b_record(
            task_slug=cand.task_slug,
            experiment=cand.experiment,
            invocation_id=f"{cand.invocation_id}_calm",
            user_prompt=str(exp_data.get("message") or ""),
            divergence=divergence_for_mb,
            tail_token_ids=calm_tail_ids,
            tail_text=calm_tail_text,
            tail_kind="calm",
            prompt_len=prompt_len,
            fallback_d=d_use,
            fallback_prefix_token_ids=prefix_use,
        )
        bad_mb = build_method_b_record(
            task_slug=cand.task_slug,
            experiment=cand.experiment,
            invocation_id=cand.invocation_id,
            user_prompt=str(exp_data.get("message") or ""),
            divergence=divergence_for_mb,
            tail_token_ids=bad_tail_ids,
            tail_text=bad_tail_text,
            tail_kind="observed",
            prompt_len=prompt_len,
            fallback_d=d_use,
            fallback_prefix_token_ids=prefix_use,
        )
        calm_run = {
            "assistant_text": calm_spec.calm_text,
            "tail_text": calm_tail_text,
            "chaos": calm_chaos,
            "invocation": {
                "id": f"{cand.invocation_id}_calm",
                "completion_tokens": len(calm_full),
                "assistant": calm_spec.calm_text,
                "chaos": calm_chaos,
            },
            "experiment": cand.experiment,
            "task_slug": cand.task_slug,
            "method_b_record": calm_mb.to_dict() if calm_mb else None,
        }
        bad_run = {
            "assistant_text": bad_text,
            "chaos": inv.get("chaos") or {},
            "invocation": inv,
            "experiment": cand.experiment,
            "task_slug": cand.task_slug,
            "method_b_record": bad_mb.to_dict() if bad_mb else None,
        }
        pair = validate_pair_fn(spec=spec, calm_run=calm_run, bad_run=bad_run, prompt_len=prompt_len)
        if not pair["pass"]:
            failed = []
            for _bucket in ("pair_checks",):
                for c in pair.get("pair_checks") or []:
                    if not c.get("pass"):
                        failed.append(c.get("id") or c)
            for arm in ("calm", "bad"):
                for c in (pair.get(arm) or {}).get("checks") or []:
                    if not c.get("pass"):
                        failed.append(f"{arm}:{c.get('id')}")
            scan.rejections.append(
                MinerRejection(
                    cand.experiment,
                    cand.invocation_id,
                    cand.task_slug,
                    "validate_pair",
                    f"failed={failed[:12]}",
                )
            )
            scan.n_rejected += 1
            continue

        slug = make_pair_slug(cand.experiment, cand.invocation_id, d_use)
        out_dir = natural_root / slug
        if mode == "dry_run":
            # Dry-run "admit" = pair validated; NOT pool-eligible capture on disk.
            scan.admitted_slugs.append(slug)
            admitted += 1
            continue

        # Skip only pool-eligible admits; retry/purge partial or failed captures.
        if _pool_eligible_on_disk(out_dir):
            continue
        if out_dir.exists():
            _purge_capture_dir(out_dir)

        if bundle is None or tokenizer is None:
            scan.rejections.append(
                MinerRejection(
                    cand.experiment, cand.invocation_id, cand.task_slug, "no_bundle", "capture requires model"
                )
            )
            scan.n_rejected += 1
            continue

        try:
            prompt_ids = resolve_prompt_ids_from_session(inv, exp_data, tokenizer)
        except ValueError as exc:
            scan.rejections.append(
                MinerRejection(cand.experiment, cand.invocation_id, cand.task_slug, "prompt_reconstruct", str(exc))
            )
            scan.n_rejected += 1
            continue

        try:
            capture = capture_matched_pair_activations(
                bundle,
                prompt_ids,
                calm_full,
                bad_ids,
                d_use,
                layers=list(PROBE_LAYERS),
            )
        except Exception as exc:  # noqa: BLE001 — soft-reject so batch continues
            _purge_capture_dir(out_dir)
            scan.rejections.append(
                MinerRejection(
                    cand.experiment,
                    cand.invocation_id,
                    cand.task_slug,
                    "capture_error",
                    f"{type(exc).__name__}: {exc}",
                )
            )
            scan.n_rejected += 1
            continue

        capture["source"] = "natural"
        capture["pair_slug"] = slug
        capture["task_slug"] = cand.task_slug

        import torch

        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            torch.save(capture, out_dir / "activations_at_d.pt")
            (out_dir / "matched_prefix_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
            (out_dir / "method_b_calm.json").write_text(
                json.dumps(calm_run["method_b_record"], indent=2), encoding="utf-8"
            )
            (out_dir / "method_b_bad.json").write_text(
                json.dumps(bad_run["method_b_record"], indent=2), encoding="utf-8"
            )

            manifest = CaptureManifest(
                source="natural",
                pair_slug=slug,
                d=d_use,
                prompt_len=prompt_len,
                layers=list(PROBE_LAYERS),
                capture_method=str(capture.get("capture_method") or "shared_kv_prefill"),
                extraction_position="divergent",
                calm_purity_pass=bool(pair["calm"]["pass"]),
                bad_purity_pass=bool(pair["bad"]["pass"]),
                pair_gate_pass=bool(pair["pass"]),
                kv_live_gate_pass=kv_ok,
                kv_live_gate_timestamp=str(kv_info.get("timestamp") or "") or None,
                notes=[f"mined from {cand.experiment}"],
            )
            cap = verify_capture_integrity(capture, manifest=manifest)
            adm = admission_checks(
                pair_validate_pass=pair["pass"],
                cap_pass=cap["pass"],
                manifest=manifest,
                capture_dir=out_dir,
            )
            write_admission_gate(
                out_dir,
                source="natural",
                pass_=adm["pass"],
                checks=adm["checks"] + cap["checks"],
                pool_eligible=adm["pool_eligible"],
                extra={"experiment": cand.experiment, "invocation_id": cand.invocation_id},
            )
            write_manifest(manifest, out_dir)
            if adm["pass"] and adm["pool_eligible"]:
                scan.admitted_slugs.append(slug)
                admitted += 1
            else:
                _purge_capture_dir(out_dir)
                scan.rejections.append(
                    MinerRejection(
                        cand.experiment,
                        cand.invocation_id,
                        cand.task_slug,
                        "admission",
                        str(adm["checks"]),
                    )
                )
                scan.n_rejected += 1
        except Exception as exc:  # noqa: BLE001 — leave no poisoned dir behind
            _purge_capture_dir(out_dir)
            scan.rejections.append(
                MinerRejection(
                    cand.experiment,
                    cand.invocation_id,
                    cand.task_slug,
                    "capture_write_error",
                    f"{type(exc).__name__}: {exc}",
                )
            )
            scan.n_rejected += 1

    scan.n_admitted = admitted
    return scan
