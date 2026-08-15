"""Capture provenance + content addressing for Method B activations (UNI_CAA01)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from jlens_qwen.chaos.constants import PROBE_LAYERS

SourceKind = Literal["practice", "natural"]
CaptureMethod = Literal["shared_kv_prefill", "full_forward", "legacy_position_index"]
ExtractionPosition = Literal["divergent", "shared"]

HASHED_FILES = (
    "activations_at_d.pt",
    "activations_at_d.json",
    "matched_prefix_spec.json",
    "admission_gate.json",
)


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class CaptureManifest:
    """Machine-checkable provenance for one Method B activation capture."""

    source: SourceKind
    pair_slug: str
    vector_id: str = "UNI_CAA01"
    d: int | None = None
    prompt_len: int | None = None
    layers: list[int] = field(default_factory=lambda: list(PROBE_LAYERS))
    capture_method: CaptureMethod = "shared_kv_prefill"
    extraction_position: ExtractionPosition = "divergent"
    calm_purity_pass: bool = False
    bad_purity_pass: bool = False
    pair_gate_pass: bool = False
    kv_live_gate_pass: bool = False
    kv_live_gate_timestamp: str | None = None
    sha256: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    path_hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaptureManifest:
        return cls(
            source=data["source"],
            pair_slug=str(data.get("pair_slug") or ""),
            vector_id=str(data.get("vector_id") or "UNI_CAA01"),
            d=data.get("d"),
            prompt_len=data.get("prompt_len"),
            layers=list(data.get("layers") or PROBE_LAYERS),
            capture_method=data.get("capture_method") or "shared_kv_prefill",
            extraction_position=data.get("extraction_position") or "divergent",
            calm_purity_pass=bool(data.get("calm_purity_pass")),
            bad_purity_pass=bool(data.get("bad_purity_pass")),
            pair_gate_pass=bool(data.get("pair_gate_pass")),
            kv_live_gate_pass=bool(data.get("kv_live_gate_pass")),
            kv_live_gate_timestamp=data.get("kv_live_gate_timestamp"),
            sha256=dict(data.get("sha256") or {}),
            notes=list(data.get("notes") or []),
            path_hint=str(data.get("path_hint") or ""),
        )

    def pool_eligible(self, *, capture_dir: Path | None = None) -> bool:
        """Natural-only; never true for practice. Requires all purity + KV live + hashes."""
        if self.source != "natural":
            return False
        if capture_dir is not None:
            path_s = str(capture_dir).replace("\\", "/")
            if "practice_pair" in path_s:
                return False
            if self.sha256:
                verify = verify_on_disk(capture_dir, self)
                if not verify["pass"]:
                    return False
        if not (
            self.calm_purity_pass
            and self.bad_purity_pass
            and self.pair_gate_pass
            and self.kv_live_gate_pass
        ):
            return False
        if list(self.layers) != list(PROBE_LAYERS):
            return False
        if self.extraction_position != "divergent":
            return False
        return True


def compute_sha256_map(capture_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in HASHED_FILES:
        digest = sha256_file(capture_dir / name)
        if digest is not None:
            out[name] = digest
    return out


def write_manifest(manifest: CaptureManifest, capture_dir: Path) -> Path:
    capture_dir = Path(capture_dir)
    capture_dir.mkdir(parents=True, exist_ok=True)
    manifest.path_hint = str(capture_dir)
    manifest.sha256 = compute_sha256_map(capture_dir)
    path = capture_dir / "capture_manifest.json"
    path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return path


def load_manifest(capture_dir: Path) -> CaptureManifest:
    path = Path(capture_dir) / "capture_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return CaptureManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))


def verify_on_disk(capture_dir: Path, manifest: CaptureManifest | None = None) -> dict[str, Any]:
    """Re-hash files; reject tampering. Returns {pass, checks}."""
    capture_dir = Path(capture_dir)
    m = manifest or load_manifest(capture_dir)
    checks: list[dict[str, Any]] = []
    current = compute_sha256_map(capture_dir)

    for name, expected in (m.sha256 or {}).items():
        got = current.get(name)
        ok = got == expected
        checks.append(
            {
                "id": f"hash_{name}",
                "pass": ok,
                "detail": f"expected={expected[:12]}… got={None if got is None else got[:12] + '…'}",
            }
        )

    # At least one activation file must be hashed when expected.
    has_act = any(k.startswith("activations_at_d") for k in (m.sha256 or {}))
    checks.append(
        {
            "id": "hash_activations_present",
            "pass": has_act or not (capture_dir / "activations_at_d.pt").exists(),
            "detail": f"hashed_keys={list(m.sha256.keys())}",
        }
    )

    path_ok = "practice_pair" not in str(capture_dir).replace("\\", "/") or m.source == "practice"
    checks.append(
        {
            "id": "path_source_coherent",
            "pass": path_ok,
            "detail": f"source={m.source} path={capture_dir}",
        }
    )

    return {"pass": all(c["pass"] for c in checks), "checks": checks, "current_sha256": current}


def write_admission_gate(
    capture_dir: Path,
    *,
    source: SourceKind,
    pass_: bool,
    checks: list[dict[str, Any]],
    pool_eligible: bool,
    extra: dict[str, Any] | None = None,
) -> Path:
    capture_dir = Path(capture_dir)
    capture_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "gate": "admission",
        "version": 1,
        "timestamp": utc_now_iso(),
        "pass": bool(pass_),
        "source": source,
        "pool_eligible": bool(pool_eligible) and source == "natural" and bool(pass_),
        "checks": checks,
    }
    if extra:
        payload.update(extra)
    path = capture_dir / "admission_gate.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def tag_practice_pair(
    practice_dir: Path,
    *,
    pair_slug: str = "practice_token_bucket",
    d: int | None = None,
    prompt_len: int | None = None,
) -> CaptureManifest:
    """Mark practice_pair as source=practice, pool_eligible=false forever."""
    practice_dir = Path(practice_dir)
    # Prefer metadata from matched_prefix_spec / result if present.
    spec_path = practice_dir / "matched_prefix_spec.json"
    if d is None and spec_path.is_file():
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        d = spec.get("d")
    result_path = practice_dir / "practice_pair_result.json"
    if prompt_len is None and result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        prompt_len = result.get("prompt_len")

    gate_path = practice_dir / "practice_pair_gate.json"
    pair_pass = False
    if gate_path.is_file():
        pair_pass = bool(json.loads(gate_path.read_text(encoding="utf-8")).get("pass"))

    manifest = CaptureManifest(
        source="practice",
        pair_slug=pair_slug,
        d=d,
        prompt_len=prompt_len,
        layers=list(PROBE_LAYERS),
        capture_method="shared_kv_prefill",
        extraction_position="divergent",
        calm_purity_pass=pair_pass,
        bad_purity_pass=pair_pass,
        pair_gate_pass=pair_pass,
        kv_live_gate_pass=False,
        notes=[
            "practice_pair: plumbing / KV fixture only — never pool_eligible",
            "kv_method_fixture_only",
        ],
    )
    assert not manifest.pool_eligible(capture_dir=practice_dir)
    write_admission_gate(
        practice_dir,
        source="practice",
        pass_=pair_pass,
        checks=[
            {
                "id": "PR1_practice_never_pool",
                "pass": True,
                "detail": "pool_eligible forced false for practice",
            }
        ],
        pool_eligible=False,
        extra={"pair_slug": pair_slug},
    )
    # Recompute hashes after admission_gate write.
    write_manifest(manifest, practice_dir)
    return manifest


def assert_not_practice_path(path: Path | str) -> None:
    s = str(path).replace("\\", "/")
    if "practice_pair" in s:
        raise ValueError(f"practice_pair path forbidden in natural pool: {path}")
