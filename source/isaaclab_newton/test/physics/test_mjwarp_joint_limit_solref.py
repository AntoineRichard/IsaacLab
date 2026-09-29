# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for authored MuJoCo joint parameters and native Newton joint-limit defaults."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonBuilderCfg, NewtonCfg, NewtonManager
from isaaclab_newton.sim.schemas import MujocoJointCfg, apply_mujoco_joint
from newton import Model, ModelBuilder
from newton.solvers import SolverMuJoCo
from newton.usd import SchemaResolverMjc, SchemaResolverNewton, SchemaResolverPhysx

from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from isaaclab.sim import SimulationCfg, build_simulation_context

# Newton's generic ``ModelBuilder.JointDofConfig`` limit-gain defaults.
_NEWTON_DEFAULT_LIMIT_KE = 1.0e4
_NEWTON_DEFAULT_LIMIT_KD = 1.0e1

_JOINT_LIMIT = 0.35
_PHYSICS_DT = 0.005

_AUTHORED_SOLREFLIMIT = (0.01, 1.0)
"""Limit ``solref`` a joint prim authors: half MuJoCo's default time constant, critically damped."""

_AUTHORED_SOLIMPLIMIT = (0.95, 0.999, 0.0001, 0.5, 2.0)
"""Limit ``solimp`` a joint prim authors, distinct from Newton's ``(0.9, 0.95, 0.001, 0.5, 2.0)``."""

_AUTHORED_JOINT_DAMPING = 0.01
"""Passive joint damping [N*m*s/rad] a joint prim authors, in MuJoCo's per-radian units."""

_AUTHORING_JOINT = "authored"
"""Name of the hinge in :func:`_usd_hinge_pair` that carries the MuJoCo joint fragment."""

_PLAIN_JOINT = "plain"
"""Name of the hinge in :func:`_usd_hinge_pair` that authors nothing, as the in-model control."""


def _usd_hinge_pair(*, author: bool = True) -> Usd.Stage:
    """Author two limit-bounded hinges in USD, one of them carrying the MuJoCo joint fragment.

    Both hinges are identical apart from the fragment, so every assertion has an
    in-model control that isolates what the authoring changed.

    Args:
        author: Whether to apply :class:`MujocoJointCfg` to the ``authored`` hinge.

    Returns:
        The in-memory stage.
    """
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.ArticulationRootAPI.Apply(UsdGeom.Xform.Define(stage, "/Articulation").GetPrim())

    links = []
    for index in range(3):
        link = UsdGeom.Cube.Define(stage, f"/Articulation/Link{index}")
        link.GetSizeAttr().Set(0.05)
        prim = link.GetPrim()
        prim.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3).Set((0.1 * index, 0.0, 0.0))
        prim.CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.TokenArray).Set(["xformOp:translate"])
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr().Set(0.05)
        mass_api.CreateDiagonalInertiaAttr().Set((2.0e-5, 2.0e-5, 2.0e-5))
        links.append(prim)

    for index, name in enumerate((_AUTHORING_JOINT, _PLAIN_JOINT), start=1):
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"/Articulation/Link{index}/{name}")
        joint.CreateBody0Rel().SetTargets([links[index - 1].GetPath()])
        joint.CreateBody1Rel().SetTargets([links[index].GetPath()])
        joint.CreateAxisAttr().Set("Z")
        joint.CreateLowerLimitAttr().Set(-np.rad2deg(_JOINT_LIMIT))
        joint.CreateUpperLimitAttr().Set(np.rad2deg(_JOINT_LIMIT))
        if name == _AUTHORING_JOINT and author:
            apply_mujoco_joint(
                MujocoJointCfg(
                    solreflimit=_AUTHORED_SOLREFLIMIT,
                    solimplimit=_AUTHORED_SOLIMPLIMIT,
                    damping=_AUTHORED_JOINT_DAMPING,
                ),
                joint.GetPrim().GetPath().pathString,
                stage,
            )
    return stage


def _import_usd_hinge_pair(*, author: bool = True) -> tuple[Model, dict[str, int]]:
    """Import :func:`_usd_hinge_pair` the way Isaac Lab imports an asset, and map hinges to DOFs.

    The MuJoCo manager passes Newton, PhysX and MuJoCo resolvers to
    ``ModelBuilder.add_usd``. Match that order so authored passive damping
    follows the same path as a spawned asset.

    Returns:
        The finalized model and a mapping from hinge name to its Newton DOF index.
    """
    builder = ModelBuilder(up_axis="Z")
    SolverMuJoCo.register_custom_attributes(builder)
    builder.add_usd(
        _usd_hinge_pair(author=author),
        schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx(), SchemaResolverMjc()],
    )
    model = builder.finalize(device="cpu")
    dof_start = model.joint_qd_start.numpy()
    dofs = {
        label.rsplit("/", maxsplit=1)[-1]: int(dof_start[joint])
        for joint, label in enumerate(model.joint_label)
        if label.rsplit("/", maxsplit=1)[-1] in (_AUTHORING_JOINT, _PLAIN_JOINT)
    }
    assert set(dofs) == {_AUTHORING_JOINT, _PLAIN_JOINT}
    return model, dofs


def _make_solver(model: Model) -> SolverMuJoCo:
    return SolverMuJoCo(model, iterations=100, ls_iterations=50, integrator="implicitfast", njmax=64)


def _mjc_joint_of_dof(solver: SolverMuJoCo, newton_dof: int) -> int:
    """Return the MuJoCo joint index a Newton DOF is mapped to."""
    dof_of_jnt = solver.mjc_jnt_to_newton_dof.numpy()[0]
    matches = [jnt for jnt in range(solver.mj_model.njnt) if int(dof_of_jnt[jnt]) == newton_dof]
    assert len(matches) == 1, f"Newton DOF {newton_dof} maps to {matches}, expected exactly one MuJoCo joint."
    return matches[0]


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


def test_usd_authored_limit_solref_and_solimp_reach_the_live_model():
    """A per-joint ``MujocoJointCfg`` lands in ``jnt_solref`` / ``jnt_solimp``, its neighbour untouched.

    This is the whole point of the fragment: the two hinges are identical apart
    from it, so the control row also pins that nothing leaks across joints.
    """
    model, dofs = _import_usd_hinge_pair()
    solver = _make_solver(model)

    solref = solver.mjw_model.jnt_solref.numpy()[0]
    solimp = solver.mjw_model.jnt_solimp.numpy()[0]
    authored_jnt = _mjc_joint_of_dof(solver, dofs[_AUTHORING_JOINT])
    plain_jnt = _mjc_joint_of_dof(solver, dofs[_PLAIN_JOINT])

    np.testing.assert_allclose(solref[authored_jnt], _AUTHORED_SOLREFLIMIT, rtol=1e-6)
    np.testing.assert_allclose(solimp[authored_jnt], _AUTHORED_SOLIMPLIMIT, rtol=1e-6)
    # The control hinge keeps the force-space conversion of Newton's generic gains.
    assert not np.allclose(solref[plain_jnt], _AUTHORED_SOLREFLIMIT)
    np.testing.assert_allclose(
        solref[plain_jnt],
        _expected_force_space_solref(solver, plain_jnt, _NEWTON_DEFAULT_LIMIT_KE, _NEWTON_DEFAULT_LIMIT_KD),
        rtol=1e-4,
    )
    assert not np.allclose(solimp[plain_jnt], _AUTHORED_SOLIMPLIMIT)


def test_usd_authored_joint_damping_reaches_the_live_model():
    """``MujocoJointCfg.damping`` lands in ``dof_damping`` verbatim, in per-radian units.

    The unauthored control is the mutation check: without the fragment the same
    hinge reads zero, so a passthrough that silently dropped the value could not
    pass. MuJoCo's ``damping`` is per radian, unlike the per-degree
    ``newton:damping`` the UsdPhysics convention scales on a revolute joint, so
    the authored number has to survive unscaled.
    """
    model, dofs = _import_usd_hinge_pair()
    unauthored, _ = _import_usd_hinge_pair(author=False)

    damping = model.joint_damping.numpy()
    assert damping[dofs[_AUTHORING_JOINT]] == pytest.approx(_AUTHORED_JOINT_DAMPING)
    assert damping[dofs[_PLAIN_JOINT]] == 0.0
    assert np.count_nonzero(unauthored.joint_damping.numpy()) == 0

    solver = _make_solver(model)
    dof_damping = solver.mjw_model.dof_damping.numpy()[0]
    # MuJoCo has its own DOF order, so the rows are reached through the joint map.
    mjc_dof = {name: int(solver.mj_model.jnt_dofadr[_mjc_joint_of_dof(solver, dof)]) for name, dof in dofs.items()}
    assert dof_damping[mjc_dof[_AUTHORING_JOINT]] == pytest.approx(_AUTHORED_JOINT_DAMPING)
    assert dof_damping[mjc_dof[_PLAIN_JOINT]] == 0.0


@pytest.mark.parametrize(
    ("field", "value", "expected_length"),
    [("solreflimit", (0.01,), 2), ("solimplimit", (0.95, 0.999, 0.0001), 5)],
)
def test_fixed_width_joint_fields_are_refused_at_the_wrong_width(field, value, expected_length):
    """A fixed-width pair authored short raises instead of reaching the joint prim.

    ``solreflimit`` and ``solimplimit`` are the two fields MuJoCo reads as arrays of a fixed width,
    and a short one is what a hand-written configuration gets wrong. Authoring it would leave the
    solver reading whatever the missing components happened to be, so the applier refuses it and the
    attribute stays unauthored.
    """
    stage = _usd_hinge_pair(author=False)
    prim_path = f"/Articulation/Link1/{_AUTHORING_JOINT}"

    with pytest.raises(ValueError, match=f"'{field}' must contain exactly {expected_length} values"):
        apply_mujoco_joint(MujocoJointCfg(**{field: value}), prim_path, stage)

    attribute = stage.GetPrimAtPath(prim_path).GetAttribute(f"mjc:{field}")
    assert not (attribute.IsValid() and attribute.HasAuthoredValue())


def test_manager_preserves_newton_force_space_joint_limits():
    """The manager leaves generic Newton joints on their native force-space defaults."""
    sim_cfg = SimulationCfg(
        dt=_PHYSICS_DT,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=64,
                nconmax=64,
                integrator="implicitfast",
            ),
            use_cuda_graph=False,
        ),
    )
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        sim._app_control_on_stop_handle = None
        builder = sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=sim.cfg.physics))
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
        sim.reset()

        solver = NewtonManager._solver
        solref = solver.mjw_model.jnt_solref.numpy()[0]
        mjc_jnt = _hinge_joints(solver)[0]
        expected = _expected_force_space_solref(solver, mjc_jnt, _NEWTON_DEFAULT_LIMIT_KE, _NEWTON_DEFAULT_LIMIT_KD)
        np.testing.assert_allclose(solref[mjc_jnt], expected, rtol=1e-4)
