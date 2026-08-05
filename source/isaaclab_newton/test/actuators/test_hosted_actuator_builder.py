# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused ABI parity tests for hosted Newton actuator construction."""

from types import SimpleNamespace

import pytest
from isaaclab_newton.actuators.adapter import _create_actuators_from_usd, _HostedActuatorRecipe
from newton.actuators import ControllerPD, Delay


def test_hosted_zero_step_delay_matches_newton_builder_rejection() -> None:
    """An authored zero-step delay fails like Newton instead of collapsing to no delay."""
    import newton

    builder = newton.ModelBuilder()
    body = builder.add_body(mass=1.0)
    joint = builder.add_joint_revolute(parent=-1, child=body)
    builder.add_articulation([joint])
    builder.add_actuator(ControllerPD, index=0, kp=2.0, kd=3.0, delay_steps=0)
    with pytest.raises(ValueError, match="max_delay must be >= 1"):
        builder.finalize(device="cpu")

    signature = (ControllerPD, True, (), ())
    recipe = _HostedActuatorRecipe(
        parsed=SimpleNamespace(controller_class=ControllerPD),
        signature=signature,
        controller_arguments={"kp": 2.0, "kd": 3.0, "const_effort": 0.0},
        component_arguments=((Delay, {"delay_steps": 0}),),
    )
    with pytest.raises(ValueError, match="max_delay must be >= 1"):
        _create_actuators_from_usd(
            stage=None,
            joint_names=["joint"],
            num_envs=1,
            num_total_joints=1,
            device="cpu",
            recipes_per_joint={0: recipe},
        )


def test_hosted_targets_keep_the_physx_wrapper_attribute_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hosted construction does not inherit Newton's model-only target-layout switch."""
    import newton

    monkeypatch.setattr(newton, "use_coord_layout_targets", True)
    signature = (ControllerPD, False, (), ())
    recipe = _HostedActuatorRecipe(
        parsed=SimpleNamespace(controller_class=ControllerPD),
        signature=signature,
        controller_arguments={"kp": 2.0, "kd": 3.0, "const_effort": 0.0},
        component_arguments=(),
    )

    hosted, _ = _create_actuators_from_usd(
        stage=None,
        joint_names=["joint"],
        num_envs=1,
        num_total_joints=1,
        device="cpu",
        recipes_per_joint={0: recipe},
    )

    assert hosted[0].control_target_pos_attr == "joint_target_pos"
    assert hosted[0].control_target_vel_attr == "joint_target_vel"


def test_hosted_duplicate_joint_recipes_keep_authored_occurrences() -> None:
    """Hosted construction retains duplicate target recipes instead of overwriting one."""
    signature = (ControllerPD, False, (), ())
    recipes = tuple(
        _HostedActuatorRecipe(
            parsed=SimpleNamespace(controller_class=ControllerPD),
            signature=signature,
            controller_arguments={"kp": stiffness, "kd": 0.0, "const_effort": 0.0},
            component_arguments=(),
        )
        for stiffness in (2.0, 7.0)
    )

    hosted, joint_signatures = _create_actuators_from_usd(
        stage=None,
        joint_names=["joint"],
        num_envs=1,
        num_total_joints=1,
        device="cpu",
        recipes_per_joint={0: recipes},
    )

    assert hosted[0].indices.numpy().tolist() == [0, 0]
    assert hosted[0].controller.kp.numpy().tolist() == [2.0, 7.0]
    assert joint_signatures == {"joint": (signature, signature)}
