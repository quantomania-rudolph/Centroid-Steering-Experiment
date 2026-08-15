"""CuratorBrief / QaBrief / gate report schemas for UNI_CAA01 calm-tail curation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Verdict = Literal["approve", "revise", "reject"]
CandidateStatus = Literal["candidate", "staged", "admitted", "rejected", "repaired"]


@dataclass
class PrefixLock:
    d: int
    prefix_token_ids: list[int]
    prefix_text: str
    immutable: bool = True


@dataclass
class DivergenceContext:
    bad_token_at_d: int | None
    bad_token_at_d_text: str
    bad_tail_summary: str
    forbidden_patterns: list[str]


@dataclass
class OutputContract:
    format: str
    target_staging_path: str
    target_admitted_path: str
    task_dir: str


@dataclass(frozen=True)
class DemoOracle:
    pair_id: str
    check_mode: Literal["eq", "truthy", "assert_true"]
    expect_repr: str


@dataclass
class CuratorBrief:
    brief_version: int
    brief_id: str
    task_slug: str
    pair_id: str
    invocation_id: str
    domain: str
    user_message: str
    panic_starter_main_py: str
    prefix_lock: PrefixLock
    divergence_context: DivergenceContext
    output_contract: OutputContract
    success_criteria: str
    task_summary: str
    provenance: dict[str, str]
    tracer_confidence: str = "medium"
    demo_oracle: DemoOracle | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QaBrief:
    pair_id: str
    d: int
    task_summary: str
    success_criteria: str
    prefix_text: str
    bad_token_at_d_text: str
    forbidden_patterns: list[str]
    extracted_main_py: str
    tail_after_prefix: str
    mechanical_gate_summary: str
    gate_failures_if_revise: str = ""
    demo_check: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.demo_check is None:
            d.pop("demo_check", None)
        return d


@dataclass
class GateCheck:
    id: str
    pass_: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "pass": self.pass_, "detail": self.detail}


@dataclass
class CuratorGateReport:
    brief_id: str
    model: str
    pass_: bool
    checks: list[GateCheck] = field(default_factory=list)
    repaired: bool = False
    calm_tail_path: str = ""
    canonical_text: str = ""
    bridge_used: bool = False
    sanitized: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "brief_id": self.brief_id,
            "model": self.model,
            "pass": self.pass_,
            "checks": [c.to_dict() for c in self.checks],
            "repaired": self.repaired,
            "calm_tail_path": self.calm_tail_path,
            "canonical_text": self.canonical_text,
            "bridge_used": self.bridge_used,
            "sanitized": self.sanitized,
        }
