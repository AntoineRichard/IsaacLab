# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Report-ready benchmark plotting with Matplotlib."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .analysis import VariantSummary

VARIANT_LABELS = {
    "kamino_current": "Kamino current",
    "kamino_pr_padmm": "PR3570 P-ADMM",
    "kamino_pr_dvi": "PR3570 tuned DVI",
    "mjwarp": "MJWarp",
    "physx": "PhysX",
}
VARIANT_COLORS = {
    "kamino_current": "#64748b",
    "kamino_pr_padmm": "#f59e0b",
    "kamino_pr_dvi": "#16a34a",
    "mjwarp": "#2563eb",
    "physx": "#9333ea",
}


def plot_learning(summaries: list[VariantSummary], output_path: Path) -> None:
    """Plot final-window learning metrics with three-seed 95% intervals."""
    tasks = list(dict.fromkeys(summary.task for summary in summaries))
    metrics = (
        ("reward", "Final-20 reward"),
        ("ep_length", "Final-20 episode length"),
        ("success_rate", "Final-20 success rate"),
    )
    figure, axes = plt.subplots(len(tasks), 3, figsize=(13.5, max(4.2, 3.8 * len(tasks))), squeeze=False)
    for task_index, task in enumerate(tasks):
        rows = [summary for summary in summaries if summary.task == task]
        labels = [VARIANT_LABELS.get(row.variant, row.variant) for row in rows]
        colors = [VARIANT_COLORS.get(row.variant, "#64748b") for row in rows]
        for axis, (field, ylabel) in zip(axes[task_index], metrics):
            estimates = [getattr(row, field) for row in rows]
            means = [estimate.mean if estimate is not None else 0.0 for estimate in estimates]
            errors = [estimate.half_width if estimate is not None else 0.0 for estimate in estimates]
            axis.bar(labels, means, yerr=errors, capsize=4, color=colors, edgecolor="white")
            for index, estimate in enumerate(estimates):
                if estimate is None:
                    axis.text(index, 0.0, "N/A", ha="center", va="bottom", fontweight="bold")
            axis.set_title(f"{task}\n{ylabel}")
            axis.set_ylabel(ylabel)
            axis.tick_params(axis="x", rotation=30)
            axis.grid(axis="y", alpha=0.25)
            axis.set_axisbelow(True)
    figure.suptitle("RSL-RL learning quality (mean ± 95% CI, three seeds)", fontweight="bold")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_runtime(summaries: list[VariantSummary], output_path: Path) -> None:
    """Plot mean steady-state iteration time with three-seed 95% intervals."""
    tasks = list(dict.fromkeys(summary.task for summary in summaries))
    figure, axes = plt.subplots(1, len(tasks), figsize=(max(5.5, 4.2 * len(tasks)), 4.4), squeeze=False)
    for axis, task in zip(axes[0], tasks):
        rows = [summary for summary in summaries if summary.task == task]
        labels = [VARIANT_LABELS.get(row.variant, row.variant) for row in rows]
        colors = [VARIANT_COLORS.get(row.variant, "#64748b") for row in rows]
        means = [row.iteration_time_s.mean for row in rows]
        errors = [row.iteration_time_s.half_width for row in rows]
        axis.bar(labels, means, yerr=errors, capsize=4, color=colors, edgecolor="white")
        axis.set_title(f"{task}\n{rows[0].num_envs} environments")
        axis.set_ylabel("Steady-state iteration time [s]")
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    figure.suptitle("RSL-RL runtime (mean ± 95% CI, three seeds)", fontweight="bold")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
