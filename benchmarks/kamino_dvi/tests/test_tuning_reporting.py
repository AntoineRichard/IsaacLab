# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the compact auditable ANYmal-D tuning addendum."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.kamino_dvi.tuning_reporting import _paginate_text, _runtime_title, write_tuning_report


def test_report_contains_required_methodology_tables_disclosures_and_figures(tmp_path: Path):
    """Markdown/PDF outputs expose metrics, coverage, failures, and provenance limits."""
    report = {
        "winner": "candidate",
        "winner_config": {"integrator": "euler"},
        "environment_count": 4096,
        "funnel": [
            {
                "stage": "Wave 1",
                "attempted_runs": 18,
                "valid_runs": 15,
                "terminal_rejected_runs": 3,
                "learning_rejected_candidates": 0,
                "promoted_candidates": 6,
            },
            {
                "stage": "Wave 2",
                "attempted_runs": 6,
                "valid_runs": 5,
                "terminal_rejected_runs": 1,
                "learning_rejected_candidates": 0,
                "promoted_candidates": 2,
                "selected_from_wave1": 3,
            },
        ],
        "runtime_rows": [{"stage": "wave1", "candidate": "candidate", "mean": 0.1, "half_width": 0.0, "n": 1}],
        "final_rows": [
            {
                "candidate": "candidate",
                "runtime": "0.100 ± 0.010",
                "reward": "10.0 ± 1.0",
                "success": "0.8 ± 0.1",
                "episode_length": "100 ± 5",
            }
        ],
        "learning_traces": [
            {
                "candidate": "candidate",
                "reward": [1, 2, 3],
                "success": [0.5, 0.6, 0.7],
                "episode_length": [100, 101, 102],
            }
        ],
        "speedups": {"clean DVI": 1.2, "legacy MJWarp": 0.8, "legacy PhysX": 0.7},
        "rejections": ["failed candidate: numerical"]
        + [f"audit rejection {index}" for index in range(180)]
        + ["FINAL-RENDERED-SENTINEL"],
        "seed_iteration_coverage": "seeds 42--44; Wave 1 40, Stage 2 100, Stage 3 300 iterations",
        "stage2_baseline_derivation": (
            "first 100 aligned iterations of clean 300-iteration baseline; final-20 is iterations 81--100"
        ),
        "legacy_limitations": "MJWarp/PhysX evidence is legacy and lacks current exact source/event provenance.",
        "bundle_git_dirty": {
            "count": 1,
            "run_ids": ["dirty-run"],
            "advisory": (
                "Broad/advisory bundle flag from plain git status, which includes untracked paths; "
                "the runner separately enforced tracked-only cleanliness before launch. A true flag "
                "does not prove that only untracked paths differed."
            ),
        },
        "derived_preflight_rejections": {
            "count": 1,
            "records": [
                {
                    "run_id": "preflight__failed_candidate__seed42__env4096__iter5__attempt0",
                    "failure": "preflight:numerical",
                    "source_head": "d" * 40,
                    "config_hash": "c" * 64,
                }
            ],
        },
    }
    paths = write_tuning_report(report, tmp_path)
    assert {path.name for path in paths} == {
        "anymal_d_dvi_tuning.md",
        "anymal_d_dvi_tuning.pdf",
        "runtime.png",
        "learning.png",
        "summary.json",
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
    json.loads(
        (tmp_path / "summary.json").read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    markdown = (tmp_path / "anymal_d_dvi_tuning.md").read_text(encoding="utf-8")
    for text in (
        "95%",
        "4096",
        "seeds 42--44",
        "iterations 81--100",
        "numerical",
        "legacy MJWarp",
        "legacy PhysX",
        "FINAL-RENDERED-SENTINEL",
        "originated in Wave 1",
        "dirty-run",
        "tracked-only cleanliness",
        "does not prove that only untracked paths differed",
        "Derived preflight",
        "preflight__failed_candidate__seed42__env4096__iter5__attempt0",
        "projected in memory",
        "Attempted runs",
        "Learning-rejected candidates",
    ):
        assert text in markdown
    assert "Final/canonical metrics use two-sided 95% Student-t confidence intervals with n=3" in markdown
    assert "Stage-2 metrics use two-sided 95% Student-t confidence intervals with n=2" not in markdown
    assert _runtime_title("completed") == "Wave 1/2 runtime ranking (single-seed observations; no CI)"


def test_early_stop_runtime_title_distinguishes_stage2_intervals():
    assert (
        _runtime_title("stopped_no_safe_finalist") == "Stage-2 runtime (95% CI, n=2) and Wave 1/2 (single seed; no CI)"
    )


def test_pdf_page_text_paginates_without_truncating_final_sentinel():
    """Long audit text remains present on a later PDF page."""
    text = "\n".join([f"audit line {index}" for index in range(180)] + ["FINAL-SENTINEL"])

    pages = _paginate_text(text, lines_per_page=40)

    assert len(pages) > 1
    assert "FINAL-SENTINEL" in pages[-1]
