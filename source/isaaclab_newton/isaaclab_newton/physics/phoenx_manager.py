# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhoenX Newton manager."""

from __future__ import annotations

from newton import Model

from .newton_manager import NewtonManager
from .phoenx_manager_cfg import PhoenXSolverCfg


class NewtonPhoenXManager(NewtonManager):
    """:class:`NewtonManager` specialization for the PhoenX solver.

    SolverPhoenX requires Newton's :class:`~newton.CollisionPipeline` with
    phoenx-specific ``contact_matching`` for warm-start. The solver attaches a
    sticky pipeline to ``model._collision_pipeline`` at construction time;
    :meth:`_initialize_contacts` adopts that pipeline onto the manager so the
    base-class contact-driver loop in
    :meth:`~NewtonManager._simulate_physics_only` runs against the same buffer
    the solver expects.
    """

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: PhoenXSolverCfg) -> None:
        """Stub — full implementation deferred to Task 2."""
        raise NotImplementedError("NewtonPhoenXManager._build_solver pending implementation in Task 2.")
