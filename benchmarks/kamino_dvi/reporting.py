# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Markdown and dependency-free Matplotlib PDF benchmark reports."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .analysis import PartialRunSummary, VariantSummary
from .plotting import VARIANT_LABELS
from .statistics import Estimate

_PARTIAL_ROWS_PER_PDF_PAGE = 20
_PARTIAL_PAGE_TITLE = "Incomplete cells: descriptive successful seeds"
_PARTIAL_PAGE_DISCLAIMER = (
    "Descriptive per-seed results from successful runs in incomplete cells. Excluded from summaries, plots, "
    "confidence intervals, and comparative speedups."
)


@dataclass(frozen=True)
class _PartialPdfPage:
    """Deterministic content for one incomplete-cell PDF page."""

    title: str
    disclaimer: str
    page_label: str
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


def _estimate(value: Estimate | None, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{value.mean:.{digits}f} ± {value.half_width:.{digits}f}"


def _key_findings(summaries: list[VariantSummary]) -> list[str]:
    """Return concise per-task runtime comparisons for the tuned DVI solver."""
    findings: list[str] = []
    for task in dict.fromkeys(summary.task for summary in summaries):
        rows = {summary.variant: summary for summary in summaries if summary.task == task}
        dvi = rows.get("kamino_pr_dvi")
        if dvi is None:
            continue
        comparisons = []
        for variant, label in (("kamino_current", "current Kamino"), ("kamino_pr_padmm", "PR3570 P-ADMM")):
            other = rows.get(variant)
            if other is not None:
                comparisons.append(
                    f"{other.iteration_time_s.mean / dvi.iteration_time_s.mean:.1f}× faster than {label}"
                )
        if comparisons:
            findings.append(f"{task}: tuned DVI is {' and '.join(comparisons)}.")
        backend_findings = []
        for variant, label in (("mjwarp", "MJWarp"), ("physx", "PhysX")):
            other = rows.get(variant)
            if other is None:
                continue
            ratio = dvi.iteration_time_s.mean / other.iteration_time_s.mean
            if 0.95 <= ratio <= 1.05:
                backend_findings.append(f"is approximately equal to {label}")
            else:
                backend_findings.append(f"remains {ratio:.1f}× slower than {label}")
        if backend_findings:
            findings.append(f"{task}: tuned DVI {' and '.join(backend_findings)}.")
    return findings


def _markdown(
    summaries: list[VariantSummary],
    issues: list[str],
    figure_paths: list[Path],
    partial_summaries: list[PartialRunSummary] | None = None,
) -> str:
    lines = [
        "# Kamino DVI Solver Benchmark",
        "",
        "RSL-RL training benchmark; values are three-seed means ± two-sided 95% Student-t confidence intervals.",
        "Steady-state runtime excludes iterations 1–10; learning metrics average the final 20 iterations.",
        "",
        "## Summary",
        "",
        "| Task | Variant | Envs | Iteration time [s] | Total FPS | Reward | Episode length | Success rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    findings = _key_findings(summaries)
    lines[5:5] = ["## Key findings", "", *(f"- {finding}" for finding in findings), ""]
    for row in summaries:
        lines.append(
            f"| {row.task} | {VARIANT_LABELS.get(row.variant, row.variant)} | {row.num_envs} | "
            f"{_estimate(row.iteration_time_s)} | {_estimate(row.total_fps, 0)} | {_estimate(row.reward)} | "
            f"{_estimate(row.ep_length)} | {_estimate(row.success_rate)} |"
        )
    if partial_summaries:
        headers, rows = _pdf_partial_table(partial_summaries)
        lines.extend(
            [
                "",
                "## Incomplete cells: descriptive successful seeds",
                "",
                "These are descriptive per-seed results from successful runs in incomplete cells. They are "
                "excluded from summaries, plots, confidence intervals, and comparative speedups.",
                "",
                f"| {' | '.join(headers)} |",
                f"|{'|'.join('---:' if index >= 2 else '---' for index in range(len(headers)))}|",
                *(f"| {' | '.join(row)} |" for row in rows),
            ]
        )
    lines.extend(["", "## Data quality and failures", ""])
    lines.extend(f"- {issue}" for issue in issues)
    if not issues:
        lines.append("- No data-quality warnings or failed runs.")
    if figure_paths:
        lines.extend(["", "## Figures", ""])
        lines.extend(f"![{path.stem}]({path.name})" for path in figure_paths)
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            "- RSL-RL, 300 training iterations, seeds 42–44.",
            "- A common environment count is selected per task from 4096 downward only after explicit capacity "
            "failures.",
            "- Current/future runner protocol: runs are sequential on one GPU and validated against exact Newton "
            "revisions and clean IsaacLab/schema ancestry for schema v1.1.",
            "- Reward and episode length use schema series; success rate uses the matching TensorBoard trace.",
        ]
    )
    return "\n".join(lines) + "\n"


def _pdf_summary_table(summaries: list[VariantSummary]) -> tuple[list[str], list[list[str]]]:
    """Return PDF summary headers and rows including all required learning metrics."""
    headers = ["Task", "Variant", "Envs", "Iteration [s]", "Total FPS", "Reward", "Episode length", "Success"]
    rows = [
        [
            row.task,
            VARIANT_LABELS.get(row.variant, row.variant),
            str(row.num_envs),
            _estimate(row.iteration_time_s),
            _estimate(row.total_fps, 0),
            _estimate(row.reward),
            _estimate(row.ep_length),
            _estimate(row.success_rate),
        ]
        for row in summaries
    ]
    return headers, rows


def _pdf_partial_table(partials: list[PartialRunSummary]) -> tuple[list[str], list[list[str]]]:
    """Return plain per-seed metrics for successful runs in incomplete cells."""
    headers = [
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
    rows = [
        [
            row.task,
            VARIANT_LABELS.get(row.variant, row.variant),
            str(row.seed),
            str(row.num_envs),
            f"{row.completed_runs}/{row.required_runs}",
            f"{row.iteration_time_s:.3f}",
            f"{row.total_fps:.0f}",
            f"{row.reward:.3f}",
            f"{row.ep_length:.3f}",
            f"{row.success_rate:.3f}" if row.success_rate is not None else "N/A",
        ]
        for row in partials
    ]
    return headers, rows


def _pdf_partial_pages(partials: list[PartialRunSummary]) -> list[_PartialPdfPage]:
    """Split incomplete-cell rows into deterministic, self-contained PDF pages."""
    headers, rows = _pdf_partial_table(partials)
    page_count = (len(rows) + _PARTIAL_ROWS_PER_PDF_PAGE - 1) // _PARTIAL_ROWS_PER_PDF_PAGE
    return [
        _PartialPdfPage(
            title=_PARTIAL_PAGE_TITLE,
            disclaimer=_PARTIAL_PAGE_DISCLAIMER,
            page_label=f"Page {page_index + 1} of {page_count}",
            headers=tuple(headers),
            rows=tuple(tuple(row) for row in rows[start : start + _PARTIAL_ROWS_PER_PDF_PAGE]),
        )
        for page_index, start in enumerate(range(0, len(rows), _PARTIAL_ROWS_PER_PDF_PAGE))
    ]


def _render_partial_pdf_page(page: _PartialPdfPage) -> plt.Figure:
    """Render one self-contained page of incomplete-cell results."""
    figure = plt.figure(figsize=(11.69, 8.27))
    figure.text(0.035, 0.955, page.title, fontsize=15, fontweight="bold", va="top")
    figure.text(0.965, 0.955, page.page_label, fontsize=8, ha="right", va="top")
    figure.text(0.035, 0.905, page.disclaimer, fontsize=8, va="top")
    axis = figure.add_axes((0.025, 0.100, 0.950, 0.730))
    axis.axis("off")
    table = axis.table(
        cellText=page.rows,
        colLabels=page.headers,
        colWidths=[0.16, 0.12, 0.045, 0.055, 0.065, 0.09, 0.075, 0.08, 0.10, 0.075],
        loc="upper center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.2)
    table.scale(1, 1.25)
    return figure


def _write_partial_pdf_pages(pdf: PdfPages, partials: list[PartialRunSummary]) -> None:
    """Render and save every deterministic incomplete-cell page."""
    for page in _pdf_partial_pages(partials):
        figure = _render_partial_pdf_page(page)
        pdf.savefig(figure)
        plt.close(figure)


def _write_pdf(
    summaries: list[VariantSummary],
    issues: list[str],
    figure_paths: list[Path],
    output_path: Path,
    partial_summaries: list[PartialRunSummary] | None = None,
) -> None:
    with PdfPages(output_path) as pdf:
        figure = plt.figure(figsize=(11.69, 8.27))
        figure.text(0.035, 0.965, "Kamino DVI Solver Benchmark", fontsize=18, fontweight="bold", va="top")
        figure.text(
            0.035,
            0.915,
            "RSL-RL · 300 iterations · three seeds · mean ± 95% Student-t CI",
            fontsize=9,
            va="top",
        )
        finding_lines = []
        for finding in _key_findings(summaries):
            finding_lines.extend(textwrap.wrap(f"• {finding}", width=145, subsequent_indent="  "))
        figure.text(0.035, 0.870, "\n".join(finding_lines), fontsize=6.8, va="top", linespacing=2.2)

        headers, rows = _pdf_summary_table(summaries)
        axis = figure.add_axes((0.025, 0.250, 0.950, 0.445))
        axis.axis("off")
        table = axis.table(
            cellText=rows,
            colLabels=headers,
            colWidths=[0.19, 0.125, 0.045, 0.115, 0.085, 0.13, 0.15, 0.13],
            loc="center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(5.6)
        table.scale(1, 1.15)

        figure.text(0.035, 0.205, "Data quality and failures", fontsize=7.2, fontweight="bold", va="top")
        warning_lines = []
        for issue in issues or ["No data-quality warnings or failed runs."]:
            warning_lines.extend(textwrap.wrap(f"• {issue}", width=175, subsequent_indent="  "))
        figure.text(0.035, 0.182, "\n".join(warning_lines), fontsize=5.3, va="top", linespacing=2.4)
        pdf.savefig(figure)
        plt.close(figure)

        if partial_summaries:
            _write_partial_pdf_pages(pdf, partial_summaries)

        for path in figure_paths:
            image = mpimg.imread(path)
            figure, axis = plt.subplots(figsize=(11.69, 8.27))
            axis.imshow(image)
            axis.axis("off")
            figure.suptitle(path.stem.replace("_", " ").title(), fontsize=15, fontweight="bold")
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)


def write_reports(
    summaries: list[VariantSummary],
    issues: list[str],
    figure_paths: list[Path],
    markdown_path: Path,
    pdf_path: Path,
    *,
    partial_summaries: list[PartialRunSummary] | None = None,
) -> None:
    """Write equivalent Markdown and PDF summaries with linked/embedded figures."""
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_markdown(summaries, issues, figure_paths, partial_summaries), encoding="utf-8")
    _write_pdf(summaries, issues, figure_paths, pdf_path, partial_summaries)
