# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MuJoCo-parity joint limits for the MuJoCo Warp backend.

Newton's :class:`~newton.ModelBuilder` gives every joint DOF the generic
force-space limit gains ``limit_ke = 1e4``, ``limit_kd = 1e1``. A USD asset that
does not author MuJoCo limit attributes inherits those defaults, and the MuJoCo
Warp solver then converts them into MuJoCo's ``jnt_solref`` pair at runtime
(``update_jnt_solref_from_invweight0_kernel``):

.. code-block:: text

    factor    = dof_invweight0 · (1 − dmax)          # dmax = jnt_solimp[1] = 0.95
    timeconst = 2 / (limit_kd · factor)
    dampratio = (limit_kd · factor) / 2 · sqrt(1 / (limit_ke · factor))
              = (limit_kd / (2 · sqrt(limit_ke))) · sqrt(factor)

With Newton's defaults this collapses to ``dampratio = 0.05 · sqrt(factor)``, so
the limit constraint is critically damped only at ``factor = 400``. Real robots
sit far from that point: a light limit-bounded servo robot (MicroDuck, effective
DOF inertia ``~2e-3 kg·m²``) has ``factor ≈ 25`` and resolves to
``jnt_solref ≈ (0.008, 0.25)`` — four times underdamped, with the time constant
below the ``2·dt`` ``refsafe`` floor (which clamps ``timeconst`` but never
``dampratio``). Every limit contact then rebounds instead of absorbing, and only
passive joint damping dissipates the pumped energy. At the damping values a
light robot actually has, the state diverges to non-finite within a few hundred
steps under untrained (flailing) actions; ten times more damping masks it.

MuJoCo's own default is the critically damped ``solreflimit = (0.02, 1.0)``, so
an MJCF-derived asset behaves differently in Isaac Lab than it does upstream.

Two routes were evaluated to restore MuJoCo semantics from inside Isaac Lab:

**Derived gains (rejected).** Solving the conversion for the MuJoCo default
gives ``limit_kd = 100 / factor`` and ``limit_ke = 2500 / factor``. ``factor``
depends on ``dof_invweight0``, which is MuJoCo's per-DOF ``mean_diag(J·M⁻¹·J')``:
it differs per joint, per world, and is recomputed by MuJoCo whenever mass,
inertia or armature change — including on every mass-randomization event, which
re-runs the conversion kernel. Static authored gains therefore cannot hold
critical damping for all joints, and would silently drift out of it under domain
randomization. The route is unsound in principle, not merely inconvenient.

**Solref passthrough (chosen).** Newton already carries the per-DOF custom
attributes ``mujoco.solreflimit`` and ``mujoco.solreflimit_mode``. Tagging a DOF
``SOLREF_MODE_RAW`` makes the conversion kernel forward the authored pair
verbatim, so ``(0.02, 1.0)`` reaches the live model exactly, stays exact across
mass randomization, and round-trips through ``save_to_mjcf`` (which
deliberately omits ``solreflimit`` for force-space joints). It is expressible
entirely on the Isaac Lab side, needs no change to the ``newton`` package, and
leaves ``joint_limit_ke``/``joint_limit_kd`` untouched, so only their MuJoCo
interpretation changes.

``SOLREF_MODE_MJCF_DEFAULT`` reaches the same live values but only while the
gains equal Newton's ``2500 / 100`` MuJoCo-default sentinels, so selecting it
would additionally require overwriting the model's force-space gains with values
the asset never authored. The raw mode is preferred for that reason.

Only DOFs that inherited Newton's generic defaults are retagged. A DOF that
authors its own limit gains (from USD drive limits, an MJCF ``solreflimit``, or
a deliberate force-space configuration) is left on whatever mode the importer
selected, so explicit authoring always wins.
"""

from __future__ import annotations

import logging

import numpy as np
from newton import Model, ModelBuilder
from newton._src.solvers.mujoco.constants import (
    DEFAULT_LIMIT_SOLREF,
    SOLREF_MODE_FORCE_SPACE,
    SOLREF_MODE_RAW,
)

logger = logging.getLogger(__name__)

_NEWTON_DEFAULT_LIMIT_KE = ModelBuilder.JointDofConfig().limit_ke
"""Newton's generic joint-limit stiffness default, i.e. "the asset authored nothing"."""

_NEWTON_DEFAULT_LIMIT_KD = ModelBuilder.JointDofConfig().limit_kd
"""Newton's generic joint-limit damping default, i.e. "the asset authored nothing"."""


def apply_mujoco_default_joint_limit_solref(model: Model) -> int:
    """Give joint DOFs with unauthored limit gains MuJoCo's default limit ``solref``.

    Retags every DOF that is still on Newton's force-space mode with its generic
    default ``limit_ke``/``limit_kd`` gains to ``SOLREF_MODE_RAW`` with
    ``solreflimit = (0.02, 1.0)``, MuJoCo's critically damped default. See the
    module docstring for the conversion math and the rejected alternatives.

    Must run after :meth:`~newton.ModelBuilder.finalize` and before the
    :class:`~newton.solvers.SolverMuJoCo` is constructed, because the solver
    resolves ``jnt_solref`` from these arrays when it compiles the model.

    Args:
        model: Finalized Newton model carrying the ``mujoco`` custom-attribute
            namespace.

    Returns:
        Number of joint DOFs that were retagged.
    """
    mujoco_attrs = getattr(model, "mujoco", None)
    solref = getattr(mujoco_attrs, "solreflimit", None)
    solref_mode = getattr(mujoco_attrs, "solreflimit_mode", None)
    if solref is None or solref_mode is None or model.joint_limit_ke is None or model.joint_limit_kd is None:
        # The MuJoCo custom attributes are only registered when the MJWarp
        # solver is active; nothing to do for any other solver.
        return 0

    mode_np = solref_mode.numpy()
    solref_np = solref.numpy()
    unauthored = (
        (mode_np == SOLREF_MODE_FORCE_SPACE)
        & np.isclose(model.joint_limit_ke.numpy(), _NEWTON_DEFAULT_LIMIT_KE, rtol=1.0e-5, atol=0.0)
        & np.isclose(model.joint_limit_kd.numpy(), _NEWTON_DEFAULT_LIMIT_KD, rtol=1.0e-5, atol=0.0)
        # A raw pair authored while the mode field was absent (legacy assets)
        # is inferred from a non-zero ``solreflimit`` by Newton's kernel.
        & ~np.any(solref_np != 0.0, axis=-1)
    )
    num_retagged = int(np.count_nonzero(unauthored))
    if num_retagged == 0:
        return 0

    mode_np[unauthored] = SOLREF_MODE_RAW
    solref_np[unauthored] = DEFAULT_LIMIT_SOLREF
    solref_mode.assign(mode_np)
    solref.assign(solref_np)
    logger.info(
        "Applied MuJoCo's default joint-limit solref %s to %d/%d joint DOFs with unauthored limit gains.",
        DEFAULT_LIMIT_SOLREF,
        num_retagged,
        mode_np.size,
    )
    return num_retagged
