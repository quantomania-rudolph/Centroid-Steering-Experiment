"""Unit test for gate runner artifact summarizer."""

from __future__ import annotations

from tests.gates.run_all import summarize_from_artifacts


def test_summarize_from_artifacts_all_pass():
    report = summarize_from_artifacts(7, start=0)
    assert report["overall_pass"] is True
    assert len(report["results"]) == 8
    assert all(r["pass"] for r in report["results"])
