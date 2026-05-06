# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Newton PhoenX physics manager."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils import configclass

from .newton_manager_cfg import NewtonSolverCfg

if TYPE_CHECKING:
    from isaaclab_newton.physics import NewtonManager


@configclass
class PhoenXSolverCfg(NewtonSolverCfg):
    """Configuration for the PhoenX solver.

    PhoenX is a maximal-coordinate rigid-body solver with PGS contact resolution
    and a TGS-soft velocity-relax pass. It always uses Newton's
    :class:`~newton.CollisionPipeline` with a phoenx-specific ``contact_matching``
    setting, attached automatically by :class:`~newton.solvers.SolverPhoenX` at
    construction time. Setting :attr:`NewtonCfg.collision_cfg` while this solver
    is selected raises ``ValueError`` from
    :meth:`NewtonPhoenXManager._build_solver` at solver construction time.

    Supported joint types: REVOLUTE, PRISMATIC, BALL, FIXED, CABLE, FREE.
    DISTANCE and D6 are not supported and raise at construction.
    """

    class_type: type[NewtonManager] | str = "{DIR}.phoenx_manager:NewtonPhoenXManager"
    """Manager class for the PhoenX solver."""

    solver_type: str = "phoenx"
    """Solver type. Can be "phoenx"."""

    substeps: int = 1
    """PhoenX internal substeps per :meth:`step` call."""

    solver_iterations: int = 8
    """PGS constraint iterations per substep."""

    velocity_iterations: int = 1
    """TGS-soft velocity-relax sweeps per substep."""

    default_friction: float = 0.5
    """Fallback contact friction [-] when neither contacts nor shape material carry one."""

    step_layout: str = "multi_world"
    """Kernel launch strategy. ``"multi_world"`` (default) for many small worlds;
    ``"single_world"`` for one or a few very large worlds."""

    threads_per_world: int | str = "auto"
    """Threads per world for ``multi_world`` fast-tail kernels.

    ``"auto"`` (default) selects per-step from the colour-size histogram.
    Integer values (``32``, ``16``, ``8``) pin the warp size."""

    max_thread_blocks: int | None = None
    """Optional cap on the persistent grid for ``single_world`` PGS sweeps.

    ``None`` (default) keeps the auto-sized grid.  Has no effect when
    :attr:`step_layout` is ``"multi_world"``."""

    velocity_readout: str = "substep_end"
    """Convention for ``state_out.body_qd``.

    One of ``"substep_end"`` (bit-faithful to PhoenX's last substep velocity),
    ``"finite_difference"`` (pose delta over the outer step), or
    ``"substep_average"`` (averaged across internal substeps).
    ``"substep_average"`` matches MuJoCo Warp's post-integration ``qvel``
    convention and is recommended for RL inference scenes."""
