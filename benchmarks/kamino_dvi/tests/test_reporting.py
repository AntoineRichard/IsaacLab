# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Markdown and PDF benchmark reports."""

from pathlib import Path

from benchmarks.kamino_dvi import reporting
from benchmarks.kamino_dvi.analysis import PartialRunSummary, VariantSummary
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
    assert "Incomplete cells: descriptive successful seeds" not in markdown.read_text(encoding="utf-8")
    assert pdf.read_bytes().startswith(b"%PDF")
    assert pdf.stat().st_size > 1000


def test_write_reports_discloses_partial_runs_as_descriptive_only(tmp_path: Path):
    """Successful seeds from incomplete cells appear separately in Markdown and PDF."""
    estimate = Estimate(1.0, 0.1, 3)
    summaries = [VariantSummary("task", "physx", 4096, estimate, estimate, estimate, estimate, estimate)]
    partials = [
        PartialRunSummary("DR Legs", "kamino_current", 42, 4096, 9.75, 10080.0, 7.25, 23.5, 0.06, 2, 3),
        PartialRunSummary("DR Legs", "kamino_current", 44, 4096, 9.72, 10100.0, 8.50, 25.0, None, 2, 3),
    ]
    markdown = tmp_path / "report.md"
    pdf = tmp_path / "report.pdf"

    write_reports(summaries, [], [], markdown, pdf, partial_summaries=partials)

    text = markdown.read_text(encoding="utf-8")
    partial_section = text.split("## Incomplete cells: descriptive successful seeds", 1)[1].split(
        "## Data quality and failures", 1
    )[0]
    assert "successful runs in incomplete cells" in partial_section
    assert "excluded from summaries, plots, confidence intervals, and comparative speedups" in partial_section
    assert "| DR Legs | Kamino current | 42 | 4096 | 2/3 | 9.750 | 10080 | 7.250 | 23.500 | 0.060 |" in partial_section
    assert "| DR Legs | Kamino current | 44 | 4096 | 2/3 | 9.720 | 10100 | 8.500 | 25.000 | N/A |" in partial_section
    assert "±" not in partial_section
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


def test_pdf_summary_table_reports_missing_success_as_na():
    """Tasks without a success definition retain their row with an explicit N/A value."""
    estimate = Estimate(1.0, 0.1, 3)
    summaries = [VariantSummary("DR Legs", "kamino_pr_dvi", 4096, estimate, estimate, estimate, estimate, None)]

    _, rows = reporting._pdf_summary_table(summaries)

    assert rows[0][-1] == "N/A"


def test_pdf_partial_table_contains_plain_per_seed_descriptive_metrics():
    """Incomplete-cell PDF rows disclose successful seeds without confidence intervals."""
    partials = [
        PartialRunSummary(
            "DR Legs",
            "kamino_current",
            42,
            4096,
            9.75,
            10080.0,
            7.25,
            23.5,
            None,
            2,
            3,
        )
    ]

    headers, rows = reporting._pdf_partial_table(partials)

    assert headers == [
        "Task",
        "Variant",
        "Seed",
        "Envs",
        "Completed",
        "Iteration [s]",
        "Total FPS",
        "Reward",
        "Episode length",
        "Success",
    ]
    assert rows == [["DR Legs", "Kamino current", "42", "4096", "2/3", "9.750", "10080", "7.250", "23.500", "N/A"]]
    assert "±" not in " ".join(rows[0])


def test_pdf_partial_pages_paginate_rows_and_repeat_context():
    """Large incomplete-cell tables produce deterministic, self-contained PDF pages."""
    partials = [
        PartialRunSummary(
            "DR Legs",
            "kamino_current",
            seed,
            4096,
            9.75,
            10080.0,
            7.25,
            23.5,
            None,
            45,
            45,
        )
        for seed in range(45)
    ]

    pages = reporting._pdf_partial_pages(partials)

    assert [len(page.rows) for page in pages] == [20, 20, 5]
    assert [page.page_label for page in pages] == ["Page 1 of 3", "Page 2 of 3", "Page 3 of 3"]
    assert all(page.title == "Incomplete cells: descriptive successful seeds" for page in pages)
    assert all(
        "Excluded from summaries, plots, confidence intervals, and comparative speedups" in page.disclaimer
        for page in pages
    )
    assert all(page.headers == pages[0].headers for page in pages)
    assert [int(row[2]) for page in pages for row in page.rows] == list(range(45))
    assert len(reporting._pdf_partial_pages(partials[:4])) == 1


def test_write_partial_pdf_pages_renders_and_saves_each_chunk(monkeypatch):
    """The PDF writer consumes every deterministic page chunk exactly once."""
    partials = [
        PartialRunSummary("DR Legs", "kamino_current", seed, 4096, 9.75, 10080.0, 7.25, 23.5, None, 45, 45)
        for seed in range(45)
    ]
    expected_pages = reporting._pdf_partial_pages(partials)
    rendered = []
    saved = []
    closed = []

    monkeypatch.setattr(reporting, "_render_partial_pdf_page", lambda page: rendered.append(page) or page)
    monkeypatch.setattr(reporting.plt, "close", closed.append)
    pdf = type("RecordingPdf", (), {"savefig": lambda _self, figure: saved.append(figure)})()

    reporting._write_partial_pdf_pages(pdf, partials)

    assert rendered == expected_pages
    assert saved == expected_pages
    assert closed == expected_pages
