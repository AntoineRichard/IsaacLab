# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless tests for Newton actuator USD authoring."""

from pxr import Sdf, Usd

from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.sim.schemas.schemas_actuators import _author_actuator_prims


def test_authoring_replaces_or_deactivates_stale_actuators_on_lab_covered_joints() -> None:
    """Implicit and explicit Lab coverage cannot leave a stale Newton actuator active."""
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/Robot", "Xform")
    joints = {
        name: stage.DefinePrim(f"/Robot/{name}", "PhysicsRevoluteJoint")
        for name in ("implicit_joint", "explicit_joint", "uncovered_joint")
    }

    def stale_actuator(name: str) -> Usd.Prim:
        actuator = stage.DefinePrim(f"/Robot/stale_{name}", "NewtonActuator")
        actuator.CreateRelationship("newton:targets").SetTargets([Sdf.Path(joints[name].GetPath())])
        return actuator

    stale_implicit = stale_actuator("implicit_joint")
    stale_explicit = stale_actuator("explicit_joint")
    stale_uncovered = stale_actuator("uncovered_joint")

    _author_actuator_prims(
        stage,
        "/Robot",
        {
            "implicit": ImplicitActuatorCfg(joint_names_expr=["implicit_joint"], stiffness=0.0, damping=0.0),
            "explicit": IdealPDActuatorCfg(
                joint_names_expr=["explicit_joint"], stiffness=2.0, damping=0.0, effort_limit=10.0
            ),
        },
    )

    assert not stale_implicit.IsActive()
    assert not stale_explicit.IsActive()
    assert stale_uncovered.IsActive()
    replacement = stage.GetPrimAtPath("/Robot/explicit_explicit_joint_actuator")
    assert replacement.IsActive()
    assert replacement.GetRelationship("newton:targets").GetTargets() == [Sdf.Path(joints["explicit_joint"].GetPath())]


def test_explicit_actuator_authoring_is_idempotent() -> None:
    """Reauthoring one explicit Lab group leaves one active, parseable Newton actuator."""
    from newton.actuators import parse_actuator_prim

    stage = Usd.Stage.CreateInMemory()
    robot = stage.DefinePrim("/Robot", "Xform")
    joint = stage.DefinePrim("/Robot/joint", "PhysicsRevoluteJoint")
    actuator_cfgs = {
        "explicit": IdealPDActuatorCfg(joint_names_expr=["joint"], stiffness=2.0, damping=0.5, effort_limit=10.0)
    }

    _author_actuator_prims(stage, "/Robot", actuator_cfgs)
    _author_actuator_prims(stage, "/Robot", actuator_cfgs)

    actuator_prims = [prim for prim in Usd.PrimRange(robot) if prim.GetTypeName() == "NewtonActuator"]
    assert len(actuator_prims) == 1
    actuator_prim = actuator_prims[0]
    assert actuator_prim.IsActive()
    parsed = parse_actuator_prim(actuator_prim)
    assert parsed is not None
    assert parsed.target_path == str(joint.GetPath())
