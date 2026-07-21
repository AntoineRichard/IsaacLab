# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :class:`isaaclab_newton.physics.KaminoSolverCfg`."""

from types import SimpleNamespace

import newton._src.solvers.kamino.config as kamino_config
import newton.solvers
import pytest
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
    assert "dvi" not in captured
    assert captured["dynamics"].preconditioning is True
    assert captured["padmm"].max_iterations == 200


def test_dvi_config_passes_dynamics_solver_to_candidate_newton(monkeypatch):
    """An explicit DVI selection must reach the candidate Newton config."""
    captured = _capture_solver_config(monkeypatch)
    monkeypatch.setattr(kamino_config, "DVISolverConfig", lambda **kwargs: SimpleNamespace(**kwargs), raising=False)

    KaminoSolverCfg(dynamics_solver="dvi").to_solver_config()

    assert captured["dynamics_solver"] == "dvi"


def test_dvi_config_passes_sparse_linear_solver_and_dvi_settings(monkeypatch):
    """An explicit DVI configuration must reach the candidate Newton sub-configs."""
    captured = _capture_solver_config(monkeypatch)

    def record_dvi_config(**kwargs: object):
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(kamino_config, "DVISolverConfig", record_dvi_config, raising=False)

    KaminoSolverCfg(
        dynamics_solver="dvi",
        sparse_jacobian=True,
        sparse_dynamics=True,
        dynamics_preconditioning=False,
        dynamics_linear_solver_type="CR",
        dynamics_linear_solver_max_iterations=9,
        dvi_block_iterations=16,
        dvi_contact_iterations=2,
    ).to_solver_config()

    assert captured["dynamics"].linear_solver_type == "CR"
    assert captured["dynamics"].linear_solver_kwargs == {"maxiter": 9}
    assert captured["dvi"].block_iterations == 16
    assert captured["dvi"].contact_iterations == 2


@pytest.mark.parametrize(
    ("configured_mode", "expected_mode"),
    [(None, "none"), ("none", "none"), ("internal", "internal"), ("containers", "containers")],
)
def test_dvi_config_normalizes_none_warmstart_mode(monkeypatch, configured_mode, expected_mode):
    """A null DVI warm-start mode must reach Newton as the literal ``"none"`` mode."""
    captured = _capture_solver_config(monkeypatch)
    monkeypatch.setattr(kamino_config, "DVISolverConfig", lambda **kwargs: SimpleNamespace(**kwargs), raising=False)

    KaminoSolverCfg(dynamics_solver="dvi", dvi_warmstart_mode=configured_mode).to_solver_config()

    assert captured["dvi"].warmstart_mode == expected_mode
