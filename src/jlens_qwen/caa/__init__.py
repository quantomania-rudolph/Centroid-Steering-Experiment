"""CAA / Method B matched-prefix utilities."""

from jlens_qwen.caa.activation_capture import (
    capture_matched_pair_activations,
    capture_residuals_at_positions,
)
from jlens_qwen.caa.admission import admission_checks, verify_capture_integrity
from jlens_qwen.caa.capture_manifest import CaptureManifest, load_manifest, verify_on_disk
from jlens_qwen.caa.method_b_databus import MethodBRecord, build_method_b_record, validate_method_b_record
from jlens_qwen.caa.steering_vectors import (
    aggregate_steering_vectors,
    compute_steering_vectors,
    discover_natural_captures,
    load_capture,
    remove_neutral_component,
    save_steering_artifacts,
)

__all__ = [
    "MethodBRecord",
    "build_method_b_record",
    "validate_method_b_record",
    "capture_residuals_at_positions",
    "capture_matched_pair_activations",
    "compute_steering_vectors",
    "aggregate_steering_vectors",
    "discover_natural_captures",
    "load_capture",
    "save_steering_artifacts",
    "remove_neutral_component",
    "CaptureManifest",
    "load_manifest",
    "verify_on_disk",
    "verify_capture_integrity",
    "admission_checks",
]
