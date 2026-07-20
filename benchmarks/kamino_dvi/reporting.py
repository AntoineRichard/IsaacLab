# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Markdown and dependency-free Matplotlib PDF benchmark reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .analysis import VariantSummary
from .plotting import VARIANT_LABELS
from .statistics import Estimate


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
        gaps = []
        for variant, label in (("mjwarp", "MJWarp"), ("physx", "PhysX")):
            other = rows.get(variant)
            if other is not None:
                gaps.append(f"{dvi.iteration_time_s.mean / other.iteration_time_s.mean:.1f}× slower than {label}")
        if gaps:
            findings.append(f"{task}: tuned DVI remains {' and '.join(gaps)}.")
    return findings


def _markdown(summaries: list[VariantSummary], issues: list[str], figure_paths: list[Path]) -> str:
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
            "- Runs are sequential on one GPU and validated against immutable IsaacLab/Newton revisions and schema "
            "v1.1.",
            "- Reward and episode length use schema series; success rate uses the matching TensorBoard trace.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_pdf(summaries: list[VariantSummary], issues: list[str], figure_paths: list[Path], output_path: Path) -> None:
    with PdfPages(output_path) as pdf:
        figure = plt.figure(figsize=(11.69, 8.27))
        figure.text(0.07, 0.93, "Kamino DVI Solver Benchmark", fontsize=22, fontweight="bold")
        figure.text(0.07, 0.88, "RSL-RL · 300 iterations · three seeds · mean ± 95% Student-t CI", fontsize=11)
        finding_text = "\n".join(f"• {finding}" for finding in _key_findings(summaries))
        figure.text(0.07, 0.80, finding_text, fontsize=9, va="top", wrap=True)
        rows = [
            [
                row.task,
                VARIANT_LABELS.get(row.variant, row.variant),
                str(row.num_envs),
                _estimate(row.iteration_time_s),
                _estimate(row.total_fps, 0),
                _estimate(row.reward),
            ]
            for row in summaries
        ]
        axis = figure.add_axes((0.04, 0.20, 0.92, 0.55))
        axis.axis("off")
        table = axis.table(
            cellText=rows,
            colLabels=["Task", "Variant", "Envs", "Iteration [s]", "Total FPS", "Reward"],
            loc="upper center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.5)
        table.scale(1, 1.35)
        warning_text = "\n".join(f"• {issue}" for issue in issues) or "• No data-quality warnings or failed runs."
        figure.text(0.07, 0.08, warning_text, fontsize=9, va="bottom", wrap=True)
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

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
) -> None:
    """Write equivalent Markdown and PDF summaries with linked/embedded figures."""
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_markdown(summaries, issues, figure_paths), encoding="utf-8")
    _write_pdf(summaries, issues, figure_paths, pdf_path)
