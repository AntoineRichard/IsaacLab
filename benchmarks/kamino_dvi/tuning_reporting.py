# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compact Markdown, plot, and PDF output for ANYmal-D DVI tuning."""

from __future__ import annotations

import json
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .manifests import write_json_atomic


def _estimate(value: Any) -> str:
    if isinstance(value, Mapping):
        return f"{float(value['mean']):.3f} ± {float(value['half_width']):.3f} (95% CI, n={int(value['n'])})"
    return str(value)


def _runtime_plot(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    ordered = sorted(rows, key=lambda row: (float(row["mean"]), str(row["candidate"])))
    figure, axis = plt.subplots(figsize=(9, max(3, 0.28 * len(ordered))))
    labels = [f"{row.get('stage', '')}: {row['candidate']}" for row in ordered]
    values = [float(row["mean"]) for row in ordered]
    errors = [float(row.get("half_width", 0.0)) for row in ordered]
    axis.barh(labels, values, xerr=errors, color="#4978a8")
    axis.invert_yaxis()
    axis.set_xlabel("Steady iteration time after iterations 1-10 [s]")
    axis.set_title("Wave 1/2 runtime ranking (single-seed observations; no confidence interval)")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _guardrail_plot(traces: Sequence[Mapping[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for trace in traces:
        label = str(trace["candidate"])
        for axis, key in zip(axes, ("reward", "success", "episode_length")):
            axis.plot(range(1, len(trace[key]) + 1), trace[key], label=label)
            axis.set_ylabel(key.replace("_", " ").title())
    axes[-1].set_xlabel("Training iteration")
    if traces:
        axes[0].legend(fontsize=7)
    figure.suptitle("Stage 2 learning guardrails (including derived clean baseline)")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# ANYmal-D Kamino DVI Tuning",
        "",
        "Final/canonical uncertainty intervals are two-sided 95% Student-t confidence intervals. "
        "Wave 1/2 rankings are single-seed observations without confidence intervals. Runtime "
        "excludes iterations 1--10; reward, TensorBoard `Metrics/success_rate`, and episode length "
        "use the final 20 iterations.",
        "",
        "## Winner and canonical gate",
        "",
        f"- Candidate: `{report['winner']}`",
        f"- Environments: {report['environment_count']}",
        f"- Resolved configuration: `{json.dumps(report['winner_config'], sort_keys=True)}`",
        f"- Canonical comparison: `{json.dumps(report.get('canonical_comparison', {}), sort_keys=True)}`",
        "",
        "## Stage funnel",
        "",
        "| Stage | Attempted | Valid | Rejected | Derived preflight | Promoted |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("funnel", []):
        lines.append(
            f"| {row['stage']} | {row['attempted']} | {row['valid']} | {row['rejected']} | "
            f"{int(row.get('derived_preflight', 0))} | {row['promoted']} |"
        )
    selected_from_wave1 = sum(int(row.get("selected_from_wave1", 0)) for row in report.get("funnel", []))
    if selected_from_wave1:
        lines.extend(
            [
                "",
                f"Stage 2 also selected {selected_from_wave1} candidate(s) that originated in Wave 1; "
                "these are not counted as Wave 2 promotions.",
            ]
        )
    lines.extend(
        [
            "",
            "## Final and canonical metrics (95% CIs)",
            "",
            "| Candidate | Runtime [s] | Reward | Success | Episode length |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in report.get("final_rows", []):
        lines.append(
            f"| {row['candidate']} | {_estimate(row['runtime'])} | {_estimate(row['reward'])} | "
            f"{_estimate(row['success'])} | {_estimate(row['episode_length'])} |"
        )
    lines.extend(["", "## Speedups", ""])
    for label, value in sorted(report.get("speedups", {}).items()):
        lines.append(f"- {label}: {float(value):.3f}×")
    lines.extend(["", "## Stability, failures, and rejections", ""])
    lines.extend(f"- {reason}" for reason in report.get("rejections", []))
    if not report.get("rejections"):
        lines.append("- No terminal failures or rejected candidates.")
    disclosure = report.get("bundle_git_dirty", {"count": 0, "run_ids": [], "advisory": ""})
    dirty_run_ids = ", ".join(str(run_id) for run_id in disclosure.get("run_ids", [])) or "none"
    preflight_disclosure = report.get("derived_preflight_rejections", {"count": 0, "records": []})
    preflight_sources = (
        ", ".join(
            f"{record['run_id']} ({record.get('failure', 'unknown')})"
            for record in preflight_disclosure.get("records", [])
        )
        or "none"
    )
    lines.extend(
        [
            "",
            "## Methodology and provenance",
            "",
            f"- Environment and coverage: {report['seed_iteration_coverage']}",
            f"- Stage 2 baseline: {report['stage2_baseline_derivation']}",
            f"- Broad bundle dirty flags: {int(disclosure.get('count', 0))}; run IDs: {dirty_run_ids}",
            f"- Bundle dirty advisory: {disclosure.get('advisory', '')}",
            f"- Derived preflight rejections: {int(preflight_disclosure.get('count', 0))}; "
            f"sources: {preflight_sources}",
            "- Failed exact seed-42 screening preflights are projected in memory as rejected Wave 1/2 "
            "records only when measured evidence is absent; measured evidence always wins.",
            f"- Legacy comparison limitation: {report['legacy_limitations']}",
            "",
            "## Figures",
            "",
            "![Runtime ranking](runtime.png)",
            "",
            "![Learning guardrails](learning.png)",
        ]
    )
    return "\n".join(lines) + "\n"


def _paginate_text(text: str, *, lines_per_page: int = 58) -> tuple[str, ...]:
    """Wrap and paginate every report line without truncation."""
    lines: list[str] = []
    for paragraph in text.splitlines():
        lines.extend(textwrap.wrap(paragraph, 145) if paragraph.strip() else [""])
    return tuple(
        "\n".join(lines[index : index + lines_per_page]) for index in range(0, len(lines), lines_per_page)
    ) or ("",)


def _pdf(markdown: str, images: Sequence[Path], path: Path) -> None:
    with PdfPages(path) as pdf:
        body = markdown.replace("#", "").replace("|", " ").replace("`", "")
        for page_index, page in enumerate(_paginate_text(body)):
            figure = plt.figure(figsize=(11.69, 8.27))
            title = "ANYmal-D Kamino DVI Tuning" if page_index == 0 else "Audit details (continued)"
            figure.text(0.04, 0.96, title, fontsize=16, fontweight="bold", va="top")
            figure.text(0.04, 0.92, page, fontsize=6.2, va="top", linespacing=1.3)
            pdf.savefig(figure)
            plt.close(figure)
        for image_path in images:
            image = mpimg.imread(image_path)
            figure, axis = plt.subplots(figsize=(11.69, 8.27))
            axis.imshow(image)
            axis.axis("off")
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)


def write_tuning_report(report: Mapping[str, Any], output_dir: Path) -> tuple[Path, ...]:
    """Write deterministic JSON, figures, Markdown, and paginated PDF output."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    runtime_path = output_dir / "runtime.png"
    learning_path = output_dir / "learning.png"
    markdown_path = output_dir / "anymal_d_dvi_tuning.md"
    pdf_path = output_dir / "anymal_d_dvi_tuning.pdf"
    _runtime_plot(report.get("runtime_rows", []), runtime_path)
    _guardrail_plot(report.get("learning_traces", []), learning_path)
    markdown = _markdown(report)
    markdown_path.write_text(markdown, encoding="utf-8")
    write_json_atomic(summary_path, report)
    _pdf(markdown, (runtime_path, learning_path), pdf_path)
    return summary_path, runtime_path, learning_path, markdown_path, pdf_path
