# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Markdown and PDF benchmark reports."""

from pathlib import Path

from benchmarks.kamino_dvi import reporting
from benchmarks.kamino_dvi.analysis import VariantSummary
from benchmarks.kamino_dvi.reporting import _key_findings, write_reports
from benchmarks.kamino_dvi.statistics import Estimate


def test_write_reports_emits_markdown_and_pdf(tmp_path: Path):
    """The report renderer produces both human-readable requested formats."""
    estimate = Estimate(1.0, 0.1, 3)
    summaries = [
        VariantSummary("task", "kamino_current", 4096, Estimate(2.0, 0.1, 3), estimate, estimate, estimate, estimate),
        VariantSummary("task", "kamino_pr_dvi", 4096, Estimate(0.4, 0.1, 3), estimate, estimate, estimate, estimate),
        VariantSummary("task", "mjwarp", 4096, Estimate(0.2, 0.1, 3), estimate, estimate, estimate, estimate),
    ]
    markdown = tmp_path / "report.md"
    pdf = tmp_path / "report.pdf"

    write_reports(summaries, ["Schema success series differs from TensorBoard."], [], markdown, pdf)

    assert "Kamino DVI Solver Benchmark" in markdown.read_text(encoding="utf-8")
    assert "4096" in markdown.read_text(encoding="utf-8")
    assert "faster than current Kamino" in markdown.read_text(encoding="utf-8")
    assert "slower than MJWarp" in markdown.read_text(encoding="utf-8")
    assert "Current/future runner protocol" in markdown.read_text(encoding="utf-8")
    assert "exact Newton revisions and clean IsaacLab/schema ancestry" in markdown.read_text(encoding="utf-8")
    assert "immutable IsaacLab/Newton revisions" not in markdown.read_text(encoding="utf-8")
    assert pdf.read_bytes().startswith(b"%PDF")
    assert pdf.stat().st_size > 1000


def test_key_findings_describes_near_equal_backend_runtime():
    """A near-unity runtime ratio is described as approximately equal, not slower."""
    estimate = Estimate(1.0, 0.1, 3)
    summaries = [
        VariantSummary("task", "kamino_pr_dvi", 4096, Estimate(1.0, 0.1, 3), estimate, estimate, estimate, estimate),
        VariantSummary("task", "physx", 4096, Estimate(1.02, 0.1, 3), estimate, estimate, estimate, estimate),
    ]

    findings = _key_findings(summaries)

    assert findings == ["task: tuned DVI is approximately equal to PhysX."]


def test_pdf_summary_table_includes_all_learning_metrics():
    """The PDF summary table contains runtime and all required learning columns."""
    estimate = Estimate(1.0, 0.1, 3)
    summaries = [VariantSummary("task", "physx", 4096, estimate, estimate, estimate, estimate, estimate)]

    headers, rows = reporting._pdf_summary_table(summaries)

    assert headers == ["Task", "Variant", "Envs", "Iteration [s]", "Total FPS", "Reward", "Episode length", "Success"]
    assert len(rows) == 1
    assert len(rows[0]) == len(headers) == 8
