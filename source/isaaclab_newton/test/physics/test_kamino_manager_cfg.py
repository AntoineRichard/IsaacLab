# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :class:`isaaclab_newton.physics.KaminoSolverCfg`."""

from types import SimpleNamespace

import newton.solvers

from isaaclab_newton.physics import KaminoSolverCfg


def _capture_solver_config(monkeypatch) -> dict[str, object]:
    """Replace Newton's top-level Kamino config constructor with a recorder."""
    captured: dict[str, object] = {}

    def record_config(**kwargs: object):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(newton.solvers, "SolverKamino", SimpleNamespace(Config=record_config))
    return captured


def test_default_config_omits_dynamics_solver_for_current_newton(monkeypatch):
    """Default construction must not pass a keyword unknown to current Newton."""
    captured = _capture_solver_config(monkeypatch)

    KaminoSolverCfg().to_solver_config()

    assert "dynamics_solver" not in captured
    assert captured["dynamics"].preconditioning is True
    assert captured["padmm"].max_iterations == 200


def test_dvi_config_passes_dynamics_solver_to_candidate_newton(monkeypatch):
    """An explicit DVI selection must reach the candidate Newton config."""
    captured = _capture_solver_config(monkeypatch)

    KaminoSolverCfg(dynamics_solver="dvi").to_solver_config()

    assert captured["dynamics_solver"] == "dvi"
