# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for MuJoCo-parity joint-limit ``solref`` on the MJWarp backend.

Covers :mod:`isaaclab_newton.physics.mjwarp_joint_limits`:

* The unauthored-gain retag lands MuJoCo's critically damped ``(0.02, 1.0)``
  limit ``solref`` in the *live* MJWarp model (the exported MJCF is not a
  semantic round trip for force-space joints, so the assertion must read the
  compiled model).
* The legacy force-space conversion, pinned against the closed-form
  ``(2 / (kd·factor), (kd/2)·sqrt(1 / (ke·factor)))`` values it produces from
  the live ``dof_invweight0`` / ``jnt_solimp``, stays reachable through the
  escape hatch.
* Joints that author explicit limit gains keep their force-space conversion.
* A light limit-bounded articulation at low joint damping diverges with the
  legacy conversion and stays finite with MuJoCo's default limit ``solref``.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonManager
from isaaclab_newton.physics.mjwarp_joint_limits import apply_mujoco_default_joint_limit_solref
from newton import Model, ModelBuilder, ModelFlags
from newton._src.solvers.mujoco.constants import (
    DEFAULT_LIMIT_SOLREF,
    SOLREF_MODE_FORCE_SPACE,
    SOLREF_MODE_RAW,
)
from newton.solvers import SolverMuJoCo

from isaaclab.sim import SimulationCfg, build_simulation_context

# Newton's generic ``ModelBuilder.JointDofConfig`` limit-gain defaults.
_NEWTON_DEFAULT_LIMIT_KE = 1.0e4
_NEWTON_DEFAULT_LIMIT_KD = 1.0e1

_NUM_LINKS = 6
_JOINT_LIMIT = 0.35
_PHYSICS_DT = 0.005


def _build_light_chain(
    device: str,
    *,
    joint_damping: float = 0.005,
    authored_gain_dofs: tuple[int, ...] = (),
) -> Model:
    """Finalize a MicroDuck-like floating-base chain of limit-bounded hinges.

    Link inertias (``2e-5 kg·m²``) and armature (``0.0018 kg·m²``) match the
    order of magnitude of a light hobby-servo robot, which is the regime where
    ``dof_invweight0`` is large and the force-space limit conversion lands far
    from critical damping.

    Args:
        device: Warp device the model is finalized on.
        joint_damping: Passive joint damping [N·m·s/rad] on every hinge.
        authored_gain_dofs: Hinge indices that author explicit (non-default)
            ``limit_ke``/``limit_kd`` gains instead of inheriting Newton's
            generic defaults.

    Returns:
        The finalized model.
    """
    builder = ModelBuilder(up_axis="Z")
    SolverMuJoCo.register_custom_attributes(builder)
    joints = []
    base = builder.add_link(mass=0.25, inertia=wp.mat33(np.diag([2.0e-4] * 3).tolist()))
    joints.append(builder.add_joint_free(child=base))
    parent = base
    for i in range(_NUM_LINKS):
        link = builder.add_link(
            xform=wp.transform(wp.vec3(0.05 * (i + 1), 0.0, 0.0), wp.quat_identity()),
            mass=0.05,
            inertia=wp.mat33(np.diag([2.0e-5] * 3).tolist()),
        )
        authored = i in authored_gain_dofs
        joints.append(
            builder.add_joint_revolute(
                parent=parent,
                child=link,
                axis=(0.0, 1.0, 0.0),
                limit_lower=-_JOINT_LIMIT,
                limit_upper=_JOINT_LIMIT,
                limit_ke=5.0e3 if authored else None,
                limit_kd=2.5e2 if authored else None,
                armature=0.0018,
                target_ke=0.0,
                target_kd=0.0,
                damping=joint_damping,
                parent_xform=wp.transform(wp.vec3(0.05, 0.0, 0.0), wp.quat_identity()),
            )
        )
        parent = link
    builder.add_articulation(joints)
    return builder.finalize(device=device)


def _make_solver(model: Model) -> SolverMuJoCo:
    return SolverMuJoCo(model, iterations=100, ls_iterations=50, integrator="implicitfast", njmax=64)


def _expected_force_space_solref(solver: SolverMuJoCo, mjc_jnt: int, ke: float, kd: float) -> tuple[float, float]:
    """Reference conversion of Newton force-space gains to MuJoCo ``solref``.

    Mirrors Newton's ``convert_solref`` with ``d_width = d_r = 1``, derived
    independently from the documented relation
    ``k_eff = k / (invweight · (1 - dmax))``.
    """
    dof_adr = int(solver.mj_model.jnt_dofadr[mjc_jnt])
    invweight = float(solver.mjw_model.dof_invweight0.numpy()[0, dof_adr])
    dmax = float(solver.mjw_model.jnt_solimp.numpy()[0, mjc_jnt][1])
    factor = invweight * (1.0 - dmax)
    return 2.0 / (kd * factor), 0.5 * kd * factor * np.sqrt(1.0 / (ke * factor))


def _hinge_joints(solver: SolverMuJoCo) -> list[int]:
    return [j for j in range(solver.mj_model.njnt) if solver.mj_model.jnt_limited[j]]


def _run_random_torques(model: Model, solver: SolverMuJoCo, steps: int, torque: float, seed: int) -> int | None:
    """Drive the hinges with random torques and report the first non-finite step."""
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    rng = np.random.default_rng(seed)
    num_dofs = model.joint_dof_count
    for step in range(steps):
        joint_f = np.zeros(num_dofs, dtype=np.float32)
        joint_f[num_dofs - _NUM_LINKS :] = rng.normal(0.0, torque, size=_NUM_LINKS)
        control.joint_f.assign(joint_f)
        solver.step(state_0, state_1, control, None, _PHYSICS_DT)
        state_0, state_1 = state_1, state_0
        if not (np.isfinite(state_0.joint_q.numpy()).all() and np.isfinite(state_0.joint_qd.numpy()).all()):
            return step
    return None


def test_force_space_conversion_underdamps_light_joint_limits():
    """Pins the legacy conversion: Newton's default gains land far from critical damping.

    This is the behavior the escape hatch restores, so the numbers are pinned
    against a reference conversion evaluated from the live model rather than
    hard-coded constants.
    """
    model = _build_light_chain("cpu")
    solver = _make_solver(model)

    solref = solver.mjw_model.jnt_solref.numpy()[0]
    modes = model.mujoco.solreflimit_mode.numpy()
    assert np.all(modes == SOLREF_MODE_FORCE_SPACE)
    for mjc_jnt in _hinge_joints(solver):
        expected = _expected_force_space_solref(solver, mjc_jnt, _NEWTON_DEFAULT_LIMIT_KE, _NEWTON_DEFAULT_LIMIT_KD)
        np.testing.assert_allclose(solref[mjc_jnt], expected, rtol=1e-4)
        # Underdamped by ~4x and stiffer than MuJoCo's 0.02 s time constant.
        assert 0.2 < solref[mjc_jnt][1] < 0.3
        assert solref[mjc_jnt][0] < 0.01


def test_default_limit_solref_is_mujoco_critically_damped():
    """Unauthored limit gains resolve to MuJoCo's ``(0.02, 1.0)`` in the live model."""
    model = _build_light_chain("cpu")
    # The free-joint DOFs carry no limit gains and are left alone.
    num_retagged = apply_mujoco_default_joint_limit_solref(model)
    assert num_retagged == _NUM_LINKS

    solver = _make_solver(model)
    solref = solver.mjw_model.jnt_solref.numpy()[0]
    for mjc_jnt in _hinge_joints(solver):
        np.testing.assert_allclose(solref[mjc_jnt], DEFAULT_LIMIT_SOLREF, rtol=1e-6)

    retagged = model.mujoco.solreflimit_mode.numpy() == SOLREF_MODE_RAW
    assert np.count_nonzero(retagged) == _NUM_LINKS
    for authored in model.mujoco.solreflimit.numpy()[retagged]:
        np.testing.assert_allclose(authored, DEFAULT_LIMIT_SOLREF, rtol=1e-6)
    # The Newton force-space gains are left untouched: only their MuJoCo
    # interpretation changes.
    ke = model.joint_limit_ke.numpy()
    assert np.allclose(ke[ke > 0.0], _NEWTON_DEFAULT_LIMIT_KE)


def test_default_limit_solref_survives_mass_randomization():
    """The retagged pair stays exact when a mass event re-runs the conversion.

    Mass randomization refreshes ``dof_invweight0`` and re-launches Newton's
    conversion kernel, which is why authored force-space gains cannot hold a
    fitted damping ratio. The raw-``solref`` route is only correct if it is
    immune to that, so pin both halves: the legacy rows drift, the retagged rows
    do not move at all.
    """
    legacy_model = _build_light_chain("cpu")
    legacy_solver = _make_solver(legacy_model)
    fixed_model = _build_light_chain("cpu")
    apply_mujoco_default_joint_limit_solref(fixed_model)
    fixed_solver = _make_solver(fixed_model)

    hinges = _hinge_joints(legacy_solver)
    legacy_before = legacy_solver.mjw_model.jnt_solref.numpy()[0].copy()
    invweight_before = legacy_solver.mjw_model.dof_invweight0.numpy()[0].copy()

    for model, solver in ((legacy_model, legacy_solver), (fixed_model, fixed_solver)):
        model.body_mass.assign(model.body_mass.numpy() * 3.0)
        model.body_inv_mass.assign(model.body_inv_mass.numpy() / 3.0)
        solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)

    # The conversion input really did change, so the pins below are meaningful.
    assert not np.allclose(legacy_solver.mjw_model.dof_invweight0.numpy()[0], invweight_before)

    legacy_after = legacy_solver.mjw_model.jnt_solref.numpy()[0]
    fixed_after = fixed_solver.mjw_model.jnt_solref.numpy()[0]
    for mjc_jnt in hinges:
        assert not np.allclose(legacy_after[mjc_jnt], legacy_before[mjc_jnt])
        np.testing.assert_array_equal(fixed_after[mjc_jnt], np.asarray(DEFAULT_LIMIT_SOLREF, dtype=np.float32))


def test_authored_limit_gains_keep_force_space_conversion():
    """DOFs that author explicit limit gains are not retagged."""
    model = _build_light_chain("cpu", authored_gain_dofs=(0, 2))
    num_retagged = apply_mujoco_default_joint_limit_solref(model)
    assert num_retagged == _NUM_LINKS - 2

    solver = _make_solver(model)
    solref = solver.mjw_model.jnt_solref.numpy()[0]
    hinges = _hinge_joints(solver)
    modes = model.mujoco.solreflimit_mode.numpy()
    dof_of_jnt = solver.mjc_jnt_to_newton_dof.numpy()[0]
    for hinge_index, mjc_jnt in enumerate(hinges):
        newton_dof = int(dof_of_jnt[mjc_jnt])
        if hinge_index in (0, 2):
            assert modes[newton_dof] == SOLREF_MODE_FORCE_SPACE
            expected = _expected_force_space_solref(solver, mjc_jnt, 5.0e3, 2.5e2)
            np.testing.assert_allclose(solref[mjc_jnt], expected, rtol=1e-4)
        else:
            assert modes[newton_dof] == SOLREF_MODE_RAW
            np.testing.assert_allclose(solref[mjc_jnt], DEFAULT_LIMIT_SOLREF, rtol=1e-6)


def test_light_articulation_stays_finite_at_low_joint_damping():
    """Regression: the underdamped limits blow up a light robot, MuJoCo's default does not.

    The failure step is nondeterministic across devices, so the budget is
    generous and the assertion is only "diverges at all" versus "never
    diverges".
    """
    legacy_model = _build_light_chain("cpu")
    legacy_failure = _run_random_torques(legacy_model, _make_solver(legacy_model), steps=600, torque=0.2, seed=0)
    assert legacy_failure is not None, "legacy force-space limits were expected to diverge"

    fixed_model = _build_light_chain("cpu")
    apply_mujoco_default_joint_limit_solref(fixed_model)
    fixed_failure = _run_random_torques(fixed_model, _make_solver(fixed_model), steps=600, torque=0.2, seed=0)
    assert fixed_failure is None


@pytest.mark.parametrize("use_mujoco_default", [True, False])
def test_manager_applies_default_limit_solref_per_cfg(use_mujoco_default):
    """End-to-end: :class:`NewtonMJWarpManager` honors the cfg escape hatch."""
    sim_cfg = SimulationCfg(
        dt=_PHYSICS_DT,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=64,
                nconmax=64,
                integrator="implicitfast",
                use_mujoco_default_joint_limit_solref=use_mujoco_default,
            ),
            use_cuda_graph=False,
        ),
    )
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        sim._app_control_on_stop_handle = None
        builder = sim.physics_manager.create_builder()
        link = builder.add_link(mass=0.05, inertia=wp.mat33(np.diag([2.0e-5] * 3).tolist()))
        joint = builder.add_joint_revolute(
            parent=-1,
            child=link,
            axis=(0.0, 0.0, 1.0),
            limit_lower=-_JOINT_LIMIT,
            limit_upper=_JOINT_LIMIT,
            armature=0.0018,
        )
        builder.add_articulation([joint])
        NewtonManager.set_builder(builder)
        sim.reset()

        solver = NewtonManager._solver
        solref = solver.mjw_model.jnt_solref.numpy()[0]
        mjc_jnt = _hinge_joints(solver)[0]
        if use_mujoco_default:
            np.testing.assert_allclose(solref[mjc_jnt], DEFAULT_LIMIT_SOLREF, rtol=1e-6)
        else:
            expected = _expected_force_space_solref(solver, mjc_jnt, _NEWTON_DEFAULT_LIMIT_KE, _NEWTON_DEFAULT_LIMIT_KD)
            np.testing.assert_allclose(solref[mjc_jnt], expected, rtol=1e-4)
