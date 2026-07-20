# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Markdown and PDF benchmark reports."""

from pathlib import Path

from benchmarks.kamino_dvi.analysis import VariantSummary
from benchmarks.kamino_dvi.reporting import write_reports
from benchmarks.kamino_dvi.statistics import Estimate


def test_write_reports_emits_markdown_and_pdf(tmp_path: Path):
    """The report renderer produces both human-readable requested formats."""
    estimate = Estimate(1.0, 0.1, 5)
    summary = VariantSummary("task", "kamino_pr_dvi", 4096, estimate, estimate, estimate, estimate, estimate)
    markdown = tmp_path / "report.md"
    pdf = tmp_path / "report.pdf"

    write_reports([summary], ["Schema success series differs from TensorBoard."], [], markdown, pdf)

    assert "Kamino DVI Solver Benchmark" in markdown.read_text(encoding="utf-8")
    assert "4096" in markdown.read_text(encoding="utf-8")
    assert pdf.read_bytes().startswith(b"%PDF")
    assert pdf.stat().st_size > 1000
