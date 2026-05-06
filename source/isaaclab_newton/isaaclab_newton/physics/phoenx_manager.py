# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhoenX Newton manager."""

from __future__ import annotations

import inspect

from newton import Model
from newton.solvers import SolverPhoenX

from isaaclab.physics import PhysicsManager

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

    Cross-config validation: rejects :attr:`NewtonCfg.collision_cfg` when set,
    because PhoenX's contact-matching requirement is not expressible through
    :class:`NewtonCollisionPipelineCfg` and overriding it would silently break
    warm-start.  Mirrors the
    :meth:`NewtonMJWarpManager._build_solver` pattern.
    """

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: PhoenXSolverCfg) -> None:
        """Construct :class:`SolverPhoenX` and populate the base-class slots.

        Filters cfg fields against the solver's ``__init__`` signature so
        non-constructor metadata (``solver_type``, ``class_type``) is not
        forwarded.
        """
        cfg = PhysicsManager._cfg
        if cfg is not None and cfg.collision_cfg is not None:
            raise ValueError(
                "NewtonCfg: collision_cfg cannot be set when solver_cfg is "
                "PhoenXSolverCfg. PhoenX attaches its own contact-matching "
                "CollisionPipeline at solver construction; remove collision_cfg."
            )

        valid = set(inspect.signature(SolverPhoenX.__init__).parameters) - {"self", "model"}
        kwargs = {k: v for k, v in solver_cfg.to_dict().items() if k in valid}
        NewtonManager._solver = SolverPhoenX(model, **kwargs)
        NewtonManager._use_single_state = False
        NewtonManager._needs_collision_pipeline = True

    @classmethod
    def _initialize_contacts(cls) -> None:
        """Adopt SolverPhoenX's auto-attached sticky :class:`CollisionPipeline`.

        SolverPhoenX's ``__init__`` writes a contact-matching pipeline to
        ``model._collision_pipeline`` when one isn't already present
        **and the model has at least one collidable shape**
        (``model.shape_count > 0``). We pull that onto the manager so the
        base-class step loop drives it rather than letting the default
        ``broad_phase="explicit"`` path construct a separate, non-matching
        pipeline that would silently break PhoenX warm-start.

        When ``model.shape_count == 0`` (no collidable shapes, e.g. in
        joint-only regression tests) the solver skips pipeline construction
        entirely; we fall back to the base-class default in that case.
        """
        pipeline = getattr(cls._model, "_collision_pipeline", None)
        if pipeline is not None:
            NewtonManager._collision_pipeline = pipeline
            if cls._contacts is None:
                NewtonManager._contacts = pipeline.contacts()
        else:
            # No collidable shapes (model.shape_count == 0) — SolverPhoenX
            # skipped pipeline construction. The base class then constructs
            # a no-op broad_phase="explicit" CollisionPipeline and allocates
            # _contacts so the step loop's _collision_pipeline.collide() call
            # in _simulate_physics_only does not crash. On empty models the
            # collide() pass is a no-op. Do NOT reach this path in production
            # scenes with shapes — that would bypass PhoenX's warm-start
            # contact_matching pipeline.
            super()._initialize_contacts()
