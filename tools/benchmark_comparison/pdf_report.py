# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate a deterministic paginated PDF from normalized benchmark outputs."""

from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
import textwrap
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import matplotlib
from matplotlib.backends.backend_pdf import PdfPages

from .manifest import RunSetManifest, validate_manifest
from .normalize import (
    FAILURE_FIELDS,
    PAIRED_SUMMARY_FIELDS,
    STARTUP_METRICS,
    expansion_orders,
    read_raw_runs_csv,
)
from .plot import PLOT_BASENAMES
from .report import ReportAudit

_REPORT_TITLE = "Isaac Lab Startup and Runtime Benchmark Report"
_PDF_METADATA = {
    "Title": _REPORT_TITLE,
    "Author": "The Isaac Lab Project Developers",
    "Creator": "Isaac Lab benchmark comparison",
    "Producer": "Isaac Lab benchmark comparison",
    "CreationDate": datetime(2026, 7, 25, tzinfo=timezone.utc),
    "ModDate": datetime(2026, 7, 25, tzinfo=timezone.utc),
}
_PDF_RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
}
_PORTRAIT_SIZE = (8.5, 11.0)
_LANDSCAPE_SIZE = (11.0, 8.5)
_METRIC_LABELS = {
    "collection_fps": "Collection FPS",
    "gpu_memory_mean_mib": "Mean GPU memory [MiB]",
    "gpu_memory_peak_mib": "Peak GPU memory [MiB]",
    "gpu_utilization_mean_pct": "Mean GPU utilization [%]",
    "gpu_utilization_sample_count": "GPU utilization samples",
    "elapsed_time_s": "Elapsed time [s]",
    "startup_total_s": "Total startup [s]",
    "startup_app_launch_s": "App launch [s]",
    "startup_python_imports_s": "Python imports [s]",
    "startup_task_config_s": "Task configuration [s]",
    "startup_env_creation_s": "Environment creation [s]",
    "startup_first_step_s": "First step [s]",
}
_METRIC_TITLES = {
    "collection_fps": "Collection FPS",
    "gpu_memory_mean_mib": "Mean GPU Memory",
    "gpu_memory_peak_mib": "Peak GPU Memory",
    "gpu_utilization_mean_pct": "Mean GPU Utilization",
    "startup_total_s": "Total Startup Time",
    "startup_phase_breakdown": "Startup Phase Breakdown",
}


def write_pdf_report(
    raw_runs_path: Path,
    paired_summary_path: Path,
    failures_path: Path,
    plot_paths: Sequence[Path],
    output_path: Path,
    *,
    manifest: RunSetManifest,
    audit: ReportAudit,
) -> Path:
    """Write the deterministic paginated benchmark report PDF.

    Args:
        raw_runs_path: Normalized successful-run CSV path.
        paired_summary_path: Normalized paired-summary CSV path.
        failures_path: Normalized failed-or-missing-attempt CSV path.
        plot_paths: Paths to the 18 required category benchmark PNG figures.
        output_path: Destination PDF path.
        manifest: Run-set identity, inventory, and matrix expansion.
        audit: Artifact counts and raw-hash integrity values.

    Returns:
        The destination PDF path after successful validation and publication.

    Raises:
        ValueError: If inputs are invalid or the rendered PDF fails validation.
        RuntimeError: If a required host PDF validation executable is unavailable.
        subprocess.CalledProcessError: If a host PDF validation command fails.
        OSError: If an input cannot be read or the PDF cannot be written or published.
    """
    expected_manifest = validate_manifest(manifest)
    runs = read_raw_runs_csv(raw_runs_path)
    summaries = _read_csv(paired_summary_path, PAIRED_SUMMARY_FIELDS)
    failures = _read_csv(failures_path, FAILURE_FIELDS)
    task_order, mode_order, _task_modes = expansion_orders(expected_manifest.expansion)
    ordered_plots = _ordered_plot_paths(plot_paths)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".pdf.tmp")
    try:
        with matplotlib.rc_context(_PDF_RCPARAMS), PdfPages(temporary, metadata=_PDF_METADATA) as pdf:
            _text_page(
                pdf,
                _REPORT_TITLE,
                (
                    "Informational paired benchmark report; it defines no performance acceptance threshold.",
                    f"Run set: {expected_manifest.run_set.value}",
                    f"Phase: {expected_manifest.phase}",
                    "Startup and runtime/resource measurements are reported separately.",
                ),
            )
            _text_page(pdf, "Methodology", _methodology_lines(expected_manifest))
            _table_pages(
                pdf,
                "Pinned revisions and execution identities",
                ("Version", "Exact Git SHA", "Environment identity"),
                (
                    (
                        "lab2",
                        expected_manifest.provenance.lab2_sha,
                        expected_manifest.provenance.environment_identity("lab2"),
                    ),
                    (
                        "lab3",
                        expected_manifest.provenance.lab3_sha,
                        expected_manifest.provenance.environment_identity("lab3"),
                    ),
                ),
                rows_per_page=24,
            )
            _table_pages(
                pdf,
                "Hardware and software inventory",
                ("Item", "Isaac Lab 2 / host", "Isaac Lab 3"),
                _inventory_rows(expected_manifest),
                rows_per_page=24,
            )
            _table_pages(
                pdf,
                "Task mapping",
                ("Logical task", "Isaac Lab 2 task", "Isaac Lab 3 task"),
                _task_mapping_rows(runs, failures, task_order),
                rows_per_page=24,
            )

            for mode in mode_order:
                _table_pages(
                    pdf,
                    f"{mode}: Startup comparison",
                    _summary_headers(),
                    _summary_rows(summaries, mode, startup=True),
                    rows_per_page=22,
                )
            for mode in mode_order:
                _table_pages(
                    pdf,
                    f"{mode}: Runtime and resource comparison",
                    _summary_headers(),
                    _summary_rows(summaries, mode, startup=False),
                    rows_per_page=22,
                )

            _table_pages(
                pdf,
                "Successful individual-run appendix",
                (
                    "Mode",
                    "Task",
                    "Ver",
                    "Seed",
                    "Startup [s]",
                    "FPS",
                    "GPU mean",
                    "GPU peak",
                    "GPU util [%]",
                    "Samples",
                    "Elapsed [s]",
                    "Attempt identity",
                ),
                tuple(
                    (
                        run.mode,
                        run.logical_task,
                        run.version,
                        str(run.seed),
                        f"{run.startup_total_s:.3f}",
                        f"{run.collection_fps:.3f}",
                        f"{run.gpu_memory_mean_mib:.3f}",
                        f"{run.gpu_memory_peak_mib:.3f}",
                        f"{run.gpu_utilization_mean_pct:.3f}",
                        str(run.gpu_utilization_sample_count),
                        f"{run.elapsed_time_s:.3f}",
                        _attempt_identity_from_artifact(run.artifact_path),
                    )
                    for run in runs
                ),
                rows_per_page=18,
            )
            _table_pages(
                pdf,
                "Failures and missing attempts",
                ("Mode", "Task", "Ver", "Seed", "Classification", "Reason", "Attempt identity"),
                _failure_rows(failures),
                rows_per_page=20,
            )
            _text_page(
                pdf,
                "Artifact integrity audit",
                (
                    f"Successful attempts: {audit.successful_attempts}",
                    f"Failed or missing attempts: {audit.failed_or_missing_attempts}",
                    f"Raw files: {audit.raw_file_count}",
                    f"Generated files: {audit.generated_file_count}",
                    f"Raw hash manifest SHA-256: {audit.raw_hash_manifest_sha256}",
                ),
            )
            for basename, path in zip(PLOT_BASENAMES, ordered_plots, strict=True):
                _plot_page(pdf, _plot_title(basename), path)

        validate_pdf(
            temporary,
            (
                expected_manifest.run_set.value,
                expected_manifest.provenance.lab2_sha,
                expected_manifest.provenance.lab3_sha,
                "Startup",
            ),
        )
        os.replace(temporary, output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return output_path


def validate_pdf(path: Path, expected_tokens: Sequence[str]) -> None:
    """Validate the PDF header, exact title, page count, and expected extracted text.

    Args:
        path: PDF path to validate.
        expected_tokens: Text tokens that must occur in the extracted PDF text.

    Raises:
        ValueError: If the PDF header, title, page count, or expected text is invalid.
        RuntimeError: If ``pdfinfo`` or ``pdftotext`` is unavailable.
        subprocess.CalledProcessError: If ``pdfinfo`` or ``pdftotext`` fails.
        OSError: If the PDF cannot be read.
    """
    if not path.read_bytes().startswith(b"%PDF-"):
        raise ValueError("report PDF header is invalid")
    for executable in ("pdfinfo", "pdftotext"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"report PDF validation requires {executable}")
    info = subprocess.run(["pdfinfo", str(path)], check=True, text=True, capture_output=True).stdout
    metadata_title = re.search(r"^Title:\s*(.*)$", info, re.MULTILINE)
    if metadata_title is None or metadata_title.group(1).strip() != _REPORT_TITLE:
        raise ValueError("report PDF metadata title is invalid")
    if not re.search(r"^Pages:\s+[1-9][0-9]*$", info, re.MULTILINE):
        raise ValueError("report PDF has no pages")
    text = subprocess.run(["pdftotext", str(path), "-"], check=True, text=True, capture_output=True).stdout
    if _REPORT_TITLE not in text:
        raise ValueError("report PDF extracted title is invalid")
    missing = [token for token in expected_tokens if token not in text]
    if missing:
        raise ValueError(f"report PDF is missing expected text: {missing}")


def _text_page(pdf: PdfPages, title: str, lines: Sequence[str]) -> None:
    """Render a portrait page with a title and wrapped text lines."""
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=_PORTRAIT_SIZE, dpi=100)
    try:
        figure.patch.set_facecolor("white")
        figure.text(0.08, 0.93, title, ha="left", va="top", fontsize=18, fontweight="bold")
        y_position = 0.86
        for line in lines:
            wrapped = textwrap.wrap(str(line), width=88, break_long_words=False, break_on_hyphens=False) or [""]
            for segment in wrapped:
                figure.text(0.08, y_position, segment, ha="left", va="top", fontsize=9)
                y_position -= 0.027
            y_position -= 0.012
        pdf.savefig(figure, dpi=100, facecolor="white", edgecolor="white")
    finally:
        plt.close(figure)


def _table_pages(
    pdf: PdfPages,
    title: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    rows_per_page: int,
) -> None:
    """Render deterministic landscape table pages with repeated headers."""
    import matplotlib.pyplot as plt

    if rows_per_page < 1:
        raise ValueError("rows_per_page must be positive")
    rendered_rows = tuple(tuple(str(cell) for cell in row) for row in rows)
    if any(len(row) != len(headers) for row in rendered_rows):
        raise ValueError("table row width does not match headers")
    chunks = tuple(
        rendered_rows[index : index + rows_per_page] for index in range(0, len(rendered_rows), rows_per_page)
    ) or ((),)
    widths = _column_widths(headers, rendered_rows)
    font_size = max(3.5, min(7.0, 45.0 / max(len(headers), 1)))
    for page_number, chunk in enumerate(chunks, start=1):
        figure, axis = plt.subplots(figsize=_LANDSCAPE_SIZE, dpi=100)
        try:
            figure.patch.set_facecolor("white")
            axis.axis("off")
            page_title = title if len(chunks) == 1 else f"{title} ({page_number}/{len(chunks)})"
            figure.suptitle(page_title, x=0.04, y=0.96, ha="left", fontsize=14, fontweight="bold")
            cell_text = chunk if chunk else (tuple(headers),)
            column_labels = tuple(headers) if chunk else None
            table = axis.table(
                cellText=cell_text,
                colLabels=column_labels,
                colWidths=widths,
                cellLoc="left",
                colLoc="left",
                loc="upper center",
                bbox=(0.015, 0.035, 0.97, 0.87),
            )
            table.auto_set_font_size(False)
            table.set_fontsize(font_size)
            for (row_index, _column_index), cell in table.get_celld().items():
                cell.set_edgecolor("#666666")
                cell.set_linewidth(0.35)
                cell.set_facecolor("#E8EEF5" if row_index == 0 else "white")
                if row_index == 0:
                    cell.get_text().set_fontweight("bold")
            pdf.savefig(figure, dpi=100, facecolor="white", edgecolor="white")
        finally:
            plt.close(figure)


def _plot_page(pdf: PdfPages, title: str, path: Path) -> None:
    """Render one PNG on a full landscape page without resampling it."""
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    try:
        image = mpimg.imread(path)
    except (OSError, SyntaxError, ValueError) as error:
        raise ValueError(f"invalid benchmark PNG: {path}") from error
    figure, axis = plt.subplots(figsize=_LANDSCAPE_SIZE, dpi=100)
    try:
        figure.patch.set_facecolor("white")
        axis.imshow(image, interpolation="none", aspect="equal")
        axis.axis("off")
        figure.suptitle(title, fontsize=14, fontweight="bold", y=0.97)
        figure.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.92)
        pdf.savefig(figure, dpi=100, facecolor="white", edgecolor="white")
    finally:
        plt.close(figure)


def _attempt_identity_from_artifact(path: str) -> str:
    """Return the attempt directory name immediately preceding the artifact leaf."""
    parts = PurePosixPath(path).parts
    if len(parts) < 2:
        raise ValueError("artifact path must contain an attempt directory and artifact leaf")
    return parts[-2]


def _read_csv(path: Path, expected_fields: tuple[str, ...]) -> tuple[dict[str, str], ...]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != expected_fields:
            raise ValueError(f"unexpected {path.name} columns: {reader.fieldnames}")
        return tuple(reader)


def _ordered_plot_paths(plot_paths: Sequence[Path]) -> tuple[Path, ...]:
    paths_by_basename: dict[str, Path] = {}
    for path in plot_paths:
        if path.suffix.lower() != ".png" or path.stem not in PLOT_BASENAMES or path.stem in paths_by_basename:
            raise ValueError(f"unexpected benchmark plot path: {path}")
        paths_by_basename[path.stem] = path
    if set(paths_by_basename) != set(PLOT_BASENAMES):
        missing = [basename for basename in PLOT_BASENAMES if basename not in paths_by_basename]
        raise ValueError(f"benchmark PDF requires exactly the 18 PNG plots; missing: {missing}")
    return tuple(paths_by_basename[basename] for basename in PLOT_BASENAMES)


def _plot_title(basename: str) -> str:
    """Return the category and metric display title for an allowed plot basename."""
    category, metric = basename.split("_", maxsplit=1)
    return f"{category.title()}: {_METRIC_TITLES[metric]}"


def _methodology_lines(manifest: RunSetManifest) -> tuple[str, ...]:
    if manifest.run_set.value == "canary":
        bounds = (
            "Both versions use PhysX, RSL-RL, 4,096 environments, and paired seed 42. Runtime modes collect "
            "10 or 25 environment steps; training runs 2 iterations."
        )
    else:
        bounds = (
            "Both versions use PhysX, RSL-RL, 4,096 environments, and paired seeds 42, 43, and 44. Runtime "
            "modes collect 100 or 1,000 environment steps; training runs 100 iterations."
        )
    return (
        bounds,
        "Only complete Lab 2/Lab 3 seed pairs contribute to paired statistics. Failures and missing attempts "
        "are not imputed. Sample standard deviations describe repeat variability.",
        "The signed delta is Isaac Lab 3 - Isaac Lab 2; the percentage delta is "
        "(Lab 3 - Lab 2) / Lab 2 x 100. A zero Lab 2 baseline is undefined.",
        "Positive collection-FPS deltas mean higher throughput; resource deltas are not labeled as inherently "
        "better or worse.",
    )


def _inventory_rows(manifest: RunSetManifest) -> tuple[tuple[str, str, str], ...]:
    host = manifest.host
    lab2 = manifest.lab2
    lab3 = manifest.lab3
    return (
        ("Hostname", host.hostname, ""),
        ("Operating system", host.os, ""),
        ("CPU", host.cpu_model, ""),
        ("Logical CPUs", str(host.logical_cpu_count), ""),
        ("GPU", host.gpu_model, ""),
        ("Physical GPU index", str(host.gpu_index) if host.gpu_index is not None else "unavailable", ""),
        ("GPU UUID", host.gpu_uuid or "unavailable", ""),
        ("NVIDIA driver", host.gpu_driver, ""),
        ("CUDA", host.cuda_version or "unavailable", ""),
        ("Isaac Lab", lab2.isaac_lab, lab3.isaac_lab),
        ("Isaac Sim", lab2.isaac_sim, lab3.isaac_sim),
        ("Python", lab2.python, lab3.python),
        ("PyTorch", lab2.pytorch, lab3.pytorch),
        ("RSL-RL", lab2.rsl_rl, lab3.rsl_rl),
        ("CPU power profile", manifest.cpu_power_profile or "unavailable", ""),
    )


def _task_mapping_rows(
    runs: Sequence[object], failures: Sequence[Mapping[str, str]], task_order: Sequence[str]
) -> tuple[tuple[str, str, str], ...]:
    mappings: dict[str, dict[str, str]] = {}
    for run in runs:
        mappings.setdefault(run.logical_task, {})[run.version] = run.concrete_task
    for failure in failures:
        logical_task = failure["logical_task"]
        if logical_task:
            mappings.setdefault(logical_task, {}).setdefault(failure["version"], failure["concrete_task"])
    order = {task: index for index, task in enumerate(task_order)}
    return tuple(
        (task, versions.get("lab2", "missing"), versions.get("lab3", "missing"))
        for task, versions in sorted(mappings.items(), key=lambda item: (order.get(item[0], len(order)), item[0]))
    )


def _summary_headers() -> tuple[str, ...]:
    return ("Task", "Metric", "Pairs", "Lab2 mean", "Lab2 std", "Lab3 mean", "Lab3 std", "Delta", "Delta [%]")


def _summary_rows(summaries: Sequence[Mapping[str, str]], mode: str, *, startup: bool) -> tuple[tuple[str, ...], ...]:
    rows: list[tuple[str, ...]] = []
    for row in summaries:
        if row["mode"] != mode or (row["metric"] in STARTUP_METRICS) != startup:
            continue
        percent_delta = (
            f"{float(row['percent_delta']):+.3f}%"
            if row["percent_delta_status"] == "available" and row["percent_delta"]
            else "undefined"
        )
        rows.append(
            (
                row["logical_task"],
                _METRIC_LABELS.get(row["metric"], row["metric"]),
                row["paired_seed_count"],
                f"{float(row['lab2_mean']):.3f}",
                f"{float(row['lab2_std']):.3f}",
                f"{float(row['lab3_mean']):.3f}",
                f"{float(row['lab3_std']):.3f}",
                f"{float(row['absolute_delta']):+.3f}",
                percent_delta,
            )
        )
    return tuple(rows)


def _failure_rows(failures: Sequence[Mapping[str, str]]) -> tuple[tuple[str, ...], ...]:
    return tuple(
        (
            row["mode"],
            row["logical_task"],
            row["version"],
            row["seed"],
            row["failure_kind"],
            row["reason"],
            _attempt_identity_from_artifact(row["artifact_path"]),
        )
        for row in failures
    )


def _column_widths(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[float]:
    weights = []
    for column_index, header in enumerate(headers):
        longest = max((len(str(row[column_index])) for row in rows), default=0)
        weights.append(float(max(5, min(32, max(len(str(header)), longest)))))
    total = sum(weights)
    return [weight / total for weight in weights]
