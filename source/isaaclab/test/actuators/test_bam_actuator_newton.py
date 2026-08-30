# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Newton-native BAM actuator component (implementation B).

The suite drives the real construction path -- author ``NewtonActuator`` prims from a
:class:`~isaaclab.actuators.BamActuatorCfg`, parse them back with
:meth:`~isaaclab.actuators.newton.NewtonActuatorAdapter.from_usd`, and step the resulting
:class:`~newton.actuators.Actuator` -- so a break anywhere between the config and the Warp
kernels shows up here. No simulator is involved: the joint state is supplied by the test.

The centrepiece is the A/B cross-check, which steps implementation A
(:class:`~isaaclab.actuators.BamActuator`) and implementation B over the same trajectory and
requires their torques to agree. That comparison is only meaningful with the solver-side
friction disabled: when MuJoCo owns the friction budget, B deliberately emits the bare motor
torque and lets the constraint solver do the clipping.
"""

import math

import numpy as np
import pytest
import torch
import warp as wp
from newton.actuators import parse_actuator_prim

from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from isaaclab.actuators import BamActuator, BamActuatorCfg, IdealPDActuatorCfg
from isaaclab.actuators.bam_model import BAM_XL330_M6_PARAMS_FILE, BamMotorParams
from isaaclab.actuators.newton import (
    BAM_CONTROL_API,
    ControllerBam,
    NewtonActuatorAdapter,
    PhysxActuatorWrapper,
    apply_bam_startup_sampling,
)
from isaaclab.sim.schemas.schemas_actuators import (
    _author_actuator_prims,
    _is_newton_native_actuator_cfg,
    _validate_newton_native_actuator_cfgs,
)
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils.types import ArticulationActions

pytestmark = pytest.mark.unit

JOINT_NAMES = ["servo_0", "servo_1"]
"""Joints of the fixture articulation; two of them so the shared-supply sag is observable."""

BACKLASH_JOINT_NAMES = ["servo_0", "passive_servo_0_backlash", "servo_1", "passive_servo_1_backlash"]
"""Joints of the serial-play fixture: each servo followed by the hinge that carries its gear play.

Interleaved, and named by the ``passive_<joint>_backlash`` convention, because that is what the
backlash asset's converter emits -- an actuator that only worked on a contiguous servo block or on
a fixed index offset would pass a tidier fixture and fail on the robot.
"""

SERVO_ONLY_EXPR = ["^(?!passive_).*"]
"""Group selection that leaves the unactuated play hinges out, as the backlash asset uses."""

SERVO_SLOTS = [index for index, name in enumerate(BACKLASH_JOINT_NAMES) if not name.startswith("passive_")]
"""Positions of the driven joints in :data:`BACKLASH_JOINT_NAMES`."""

PLAY_SLOTS = [index for index, name in enumerate(BACKLASH_JOINT_NAMES) if name.startswith("passive_")]
"""Positions of the play hinges in :data:`BACKLASH_JOINT_NAMES`."""

PLAY_LIMIT = math.radians(1.0)
"""Half the gear play of one servo [rad], i.e. the reference plant's per-side backlash."""

DT = 1.0 / 120.0
"""Physics timestep the actuators are stepped at [s]."""

VIN = 7.4
"""Supply voltage the fixture is configured with [V]."""

KP_FW = 200.0
"""Firmware proportional gain the fixture is configured with [-]."""

CROSS_CHECK_STEPS = 50
"""Steps of the A/B cross-check, short enough to stay clear of integration drift."""

CROSS_CHECK_TOLERANCE = 1e-4
"""Accepted per-step torque difference between implementations A and B [N.m]."""


def _make_cfg(**overrides) -> BamActuatorCfg:
    """Build the BAM config the fixture articulation is authored from."""
    kwargs = {"joint_names_expr": [".*"], "vin": VIN, "kp_fw": KP_FW, "dt": DT}
    kwargs.update(overrides)
    return BamActuatorCfg(**kwargs)


def _make_stage(cfg: BamActuatorCfg, joint_names: list[str] = JOINT_NAMES) -> Usd.Stage:
    """Author an articulation over *joint_names*, driven by one BAM actuator group.

    The group covers whichever of the joints its ``joint_names_expr`` selects, so a fixture
    can carry joints no actuator drives -- which is what a play hinge is.
    """
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/Robot")
    for index, name in enumerate(joint_names):
        body = UsdGeom.Xform.Define(stage, f"/World/Robot/body_{index}")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"/World/Robot/{name}")
        joint.CreateBody1Rel().SetTargets([body.GetPath()])
    _author_actuator_prims(stage, "/World/Robot", {"servo": cfg})
    return stage


def _make_adapter(
    cfg: BamActuatorCfg, num_envs: int, device: str, joint_names: list[str] = JOINT_NAMES
) -> NewtonActuatorAdapter:
    """Author, parse and build the Newton actuator adapter for the fixture."""
    return NewtonActuatorAdapter.from_usd(
        stage=_make_stage(cfg, joint_names),
        joint_names=joint_names,
        num_envs=num_envs,
        num_joints=len(joint_names),
        device=device,
        articulation_prim_path="/World/Robot",
    )


class _Harness:
    """Steps one Newton actuator over test-supplied joint state.

    Wraps the flat ``sim_state`` / ``sim_control`` pair
    :meth:`~newton.actuators.Actuator.step` expects, so a test can drive the actuator with an
    arbitrary trajectory and read back the effort it asks the solver to apply.
    """

    def __init__(self, cfg: BamActuatorCfg, num_envs: int, device: str, joint_names: list[str] = JOINT_NAMES):
        self.adapter = _make_adapter(cfg, num_envs, device, joint_names)
        assert len(self.adapter.actuators) == 1, "the fixture's joints must merge into one actuator"
        self.actuator = self.adapter.actuators[0]
        self.controller: ControllerBam = self.actuator.controller
        self.num_envs = num_envs
        self.device = device
        self.joint_names = joint_names
        shape = (num_envs, len(joint_names))
        self.state = PhysxActuatorWrapper.create(*shape, device)
        self.control = PhysxActuatorWrapper.create(*shape, device)
        self.joint_pos = wp.zeros(shape, dtype=wp.float32, device=device)
        self.joint_vel = wp.zeros(shape, dtype=wp.float32, device=device)
        self.target_pos = wp.zeros(shape, dtype=wp.float32, device=device)
        self.state.joint_q = self.joint_pos.reshape(-1)
        self.state.joint_qd = self.joint_vel.reshape(-1)
        self.control.joint_target_pos = self.target_pos.reshape(-1)
        self.control.joint_target_vel = wp.zeros(num_envs * len(joint_names), dtype=wp.float32, device=device)
        self.control.joint_act = None
        self.adapter.finalize(self.control)

    def step(self, joint_pos: np.ndarray, joint_vel: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        """Run one actuator step and return the applied effort [N.m], shape ``(num_envs, J)``."""
        self.joint_pos.assign(np.ascontiguousarray(joint_pos, dtype=np.float32))
        self.joint_vel.assign(np.ascontiguousarray(joint_vel, dtype=np.float32))
        self.target_pos.assign(np.ascontiguousarray(target_pos, dtype=np.float32))
        # The adapter's own helper kernels take the ambient Warp device, exactly as the
        # backends that scope one around the stepping loop do.
        with wp.ScopedDevice(self.device):
            self.adapter.step(self.state, self.control, DT)
        return self.control.joint_f_2d.numpy().copy()

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset the actuator state of the given environments."""
        with wp.ScopedDevice(self.device):
            self.adapter.reset(env_ids)


"""
Configuration and authoring.
"""


def test_bam_cfg_is_accepted_by_newton_native_validation():
    """The BAM config must pass the gate that ``use_newton_actuators=True`` runs."""
    cfg = _make_cfg()
    assert _is_newton_native_actuator_cfg(cfg)
    _validate_newton_native_actuator_cfgs({"servo": cfg})


def test_bam_cfg_is_rejected_on_a_host_adapter_backend():
    """A backend without an in-solver actuator path must refuse the BAM config, loudly.

    The model is written in terms of solver quantities: it publishes its friction budget into
    the solver's joint dry friction and reads the external load back out of the solver's
    generalized forces. A backend that steps native actuators through the shared host adapter
    (PhysX, OVPhysX) provides neither, so the controller would silently fall back to
    implementation A's torque-level clip *and* skip the start-up randomization the Isaac Lab
    model draws on every backend. Failing the gate instead names the one-line fix.
    """
    with pytest.raises(ValueError, match="requires the Newton backend"):
        _validate_newton_native_actuator_cfgs({"servo": _make_cfg()}, host_adapter=True)

    # The restriction is BAM's alone -- every other supported config still runs there, so the
    # flag cannot be passing by rejecting the whole native path.
    _validate_newton_native_actuator_cfgs({"legs": IdealPDActuatorCfg(joint_names_expr=[".*"])}, host_adapter=True)


def test_authored_prim_resolves_to_the_bam_controller():
    """Authoring a BAM group must produce a parseable ``NewtonBamControlAPI`` actuator prim."""
    cfg = _make_cfg(vin_min=6.0, min_delay=1, max_delay=3, delay_hold_prob=0.25, delay_update_period=4)
    stage = _make_stage(cfg)

    parsed = [p for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Robot")) if (p := parse_actuator_prim(prim))]
    assert len(parsed) == len(JOINT_NAMES)
    for entry in parsed:
        assert entry.controller_class is ControllerBam
        assert entry.component_specs == [], "the BAM delay is controller-internal, not a Delay component"
        resolved = ControllerBam.resolve_arguments(dict(entry.controller_kwargs))
        params = BamMotorParams.from_json(BAM_XL330_M6_PARAMS_FILE)
        # Deployment settings come from the config, identified constants from the fit file.
        assert resolved["kp_fw"] == pytest.approx(KP_FW)
        assert resolved["vin"] == pytest.approx(VIN)
        assert resolved["vin_min"] == pytest.approx(6.0)
        assert resolved["kt"] == pytest.approx(params.kt)
        assert resolved["armature"] == pytest.approx(params.armature)
        assert resolved["load_friction_external_quad"] == pytest.approx(params.load_friction_external_quad)
        assert (resolved["min_delay"], resolved["max_delay"]) == (1, 3)
        assert resolved["delay_hold_prob"] == pytest.approx(0.25)
        assert resolved["delay_update_period"] == 4
        assert (resolved["stribeck"], resolved["load_dependent"], resolved["quadratic"]) == (1, 1, 1)

    # The controller schema token is applied on the prim, not just implied by the parse.
    # ``NewtonBamControlAPI`` has no registered USD schema definition, so the composed
    # ``GetAppliedSchemas`` filters it out; read the authored opinion instead.
    spec = stage.GetRootLayer().GetPrimAtPath(f"/World/Robot/servo_{JOINT_NAMES[0]}_actuator")
    assert BAM_CONTROL_API in spec.GetInfo("apiSchemas").prependedItems


def test_effort_limit_is_authored_on_the_controller_not_as_a_clamping_component():
    """A BAM actuator prim must carry no USD-registered API schema beside the BAM token.

    Newton resolves an actuator prim's components from ``Usd.Prim.GetAppliedSchemas``, falling
    back to the raw ``apiSchemas`` metadata *only when that comes back empty*.
    ``NewtonBamControlAPI`` has no registered schema definition, so USD drops it from the
    composed list; a registered sibling such as ``NewtonMaxEffortClampingAPI`` would make the
    composed list non-empty and the BAM controller would vanish from the parse. The effort
    limit is therefore a controller parameter, and this test is what stops it going back.
    """
    cfg = _make_cfg(actuator_effort_limit=0.05)
    stage = _make_stage(cfg)

    prim = stage.GetPrimAtPath(f"/World/Robot/servo_{JOINT_NAMES[0]}_actuator")
    parsed = parse_actuator_prim(prim)
    assert parsed is not None and parsed.controller_class is ControllerBam
    assert parsed.component_specs == [], "a BAM prim must compose no clamping or delay component"
    assert ControllerBam.resolve_arguments(dict(parsed.controller_kwargs))["max_effort"] == pytest.approx(0.05)

    spec = stage.GetRootLayer().GetPrimAtPath(prim.GetPath())
    assert list(spec.GetInfo("apiSchemas").prependedItems) == [BAM_CONTROL_API]


def test_driven_joints_are_seeded_with_a_positive_friction():
    """MuJoCo only builds a DOF-friction row for joints whose frictionloss is positive.

    The row has to exist from the first solve -- the constraint budget is sized from the model
    as spawned -- so authoring seeds the driven joints with the budget's own floor.
    """
    stage = _make_stage(_make_cfg())
    floor = BamMotorParams.from_json(BAM_XL330_M6_PARAMS_FILE).friction_base
    for name in JOINT_NAMES:
        friction = stage.GetPrimAtPath(f"/World/Robot/{name}").GetAttribute("newton:friction")
        assert friction.IsValid() and friction.Get() == pytest.approx(floor)


def test_authoring_preserves_a_task_authored_joint_friction():
    """A joint friction the asset already carries must win over the seed."""
    cfg = _make_cfg()
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/Robot")
    for index, name in enumerate(JOINT_NAMES):
        body = UsdGeom.Xform.Define(stage, f"/World/Robot/body_{index}")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"/World/Robot/{name}")
        joint.CreateBody1Rel().SetTargets([body.GetPath()])
        joint.GetPrim().CreateAttribute("newton:friction", Sdf.ValueTypeNames.Float).Set(0.5)
    _author_actuator_prims(stage, "/World/Robot", {"servo": cfg})
    for name in JOINT_NAMES:
        assert stage.GetPrimAtPath(f"/World/Robot/{name}").GetAttribute("newton:friction").Get() == pytest.approx(0.5)


"""
Kernel behaviour.
"""


@pytest.mark.parametrize("device", test_devices())
def test_controller_reproduces_the_math_core(device):
    """One step of the Warp controller must equal the torch math core term for term."""
    from isaaclab.actuators.bam_model import (
        apply_stiction_clip,
        battery_sag,
        compute_duty,
        compute_friction_budget,
        compute_motor_torque,
        compute_stribeck_coeff,
    )

    harness = _Harness(_make_cfg(), num_envs=1, device=device)
    params = BamMotorParams.from_json(BAM_XL330_M6_PARAMS_FILE)
    rng = np.random.default_rng(0)
    pos = rng.uniform(-0.4, 0.4, (1, len(JOINT_NAMES)))
    vel = rng.uniform(-2.0, 2.0, (1, len(JOINT_NAMES)))
    target = rng.uniform(-0.4, 0.4, (1, len(JOINT_NAMES)))

    effort = harness.step(pos, vel, target)

    t_pos, t_vel, t_target = (torch.tensor(a, dtype=torch.float32) for a in (pos, vel, target))
    zeros = torch.zeros_like(t_pos)
    effective_vin = battery_sag(torch.full((1, 1), VIN), zeros, torch.zeros(1, 1), None)
    duty = compute_duty(t_target, t_pos, t_vel, torch.full((1, 1), KP_FW), effective_vin, params)
    motor = compute_motor_torque(duty, t_vel, effective_vin, params)
    # First step after construction: the velocity cache is seeded, so the estimated
    # acceleration is zero and no torque was applied yet.
    external = torch.zeros_like(t_pos)
    budget = compute_friction_budget(zeros, external, compute_stribeck_coeff(t_vel, params), params, 1.0)
    expected = apply_stiction_clip(motor, external, t_vel, budget, params.friction_viscous, DT, params.armature)

    np.testing.assert_allclose(effort, expected.numpy(), atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(
        harness.controller.friction_budget.numpy(), budget.numpy().reshape(-1), atol=1e-6, rtol=0.0
    )
    np.testing.assert_allclose(harness.controller.motor_torque.numpy(), motor.numpy().reshape(-1), atol=1e-6, rtol=0.0)


@pytest.mark.parametrize("effort_limit", [None, 0.05])
@pytest.mark.parametrize("device", test_devices())
def test_cross_check_against_the_lab_executed_actuator(device, effort_limit):
    """Implementations A and B must apply the same torque on the same trajectory.

    Both are driven by the *identical* recorded state sequence -- a pendulum swinging under
    A's own efforts -- with the command delay off and the friction scale at one, so every
    stage of the pipeline (supply sag, firmware law, DC-motor equation, external-torque
    estimate, friction budget and stiction clip) is compared at once. The solver-side friction
    is deliberately not engaged here: with MuJoCo owning the budget, B emits the bare motor
    torque and the comparison would be against a different quantity.

    The parameterization matters beyond the clamp itself. A caches the *clipped* effort and
    subtracts it in its external-torque estimate, while a Newton controller runs before the
    clamping stage, so B has to read the applied effort back after the fact; a limit low enough
    to bite is what pins that.
    """
    cfg = _make_cfg(actuator_effort_limit=effort_limit)
    lab_actuator = BamActuator(
        cfg=cfg,
        joint_names=list(JOINT_NAMES),
        joint_ids=slice(None),
        num_envs=1,
        device=device,
        actuator_effort_limit=effort_limit,
        actuator_velocity_limit=None,
    )
    harness = _Harness(cfg, num_envs=1, device=device)
    assert not harness.controller.solver_applies_friction

    # A one-joint-per-environment pendulum integrated from A's own efforts: the trajectory
    # only has to be a realistic, non-trivial excitation shared by both implementations.
    inertia, load = 2.0e-3, 9.81e-3
    position = np.array([[0.3, -0.2]])
    velocity = np.zeros((1, len(JOINT_NAMES)))
    target = np.zeros((1, len(JOINT_NAMES)))

    peak_motor_torque = 0.0
    for step in range(CROSS_CHECK_STEPS):
        native = harness.step(position, velocity, target)
        peak_motor_torque = max(peak_motor_torque, float(np.abs(harness.controller.motor_torque.numpy()).max()))
        lab = lab_actuator.compute(
            ArticulationActions(joint_positions=torch.tensor(target, dtype=torch.float32, device=device)),
            torch.tensor(position, dtype=torch.float32, device=device),
            torch.tensor(velocity, dtype=torch.float32, device=device),
        ).joint_efforts
        lab_effort = lab.detach().cpu().numpy()
        np.testing.assert_allclose(
            native, lab_effort, atol=CROSS_CHECK_TOLERANCE, rtol=0.0, err_msg=f"torques diverge at step {step}"
        )
        acceleration = (lab_effort + load * np.cos(position)) / inertia
        velocity = velocity + acceleration * DT
        position = position + velocity * DT

    # A non-trivial excitation, not a pair of dead actuators agreeing on zero.
    assert np.abs(lab_effort).max() > 1e-3
    if effort_limit is not None:
        assert peak_motor_torque > effort_limit, "the effort limit never bit, so the clamp was not exercised"
        assert np.abs(native).max() <= effort_limit + 1e-6, "the emitted torque escaped the effort limit"


@pytest.mark.parametrize("device", test_devices())
def test_solver_mode_emits_the_motor_torque_and_publishes_the_budget(device):
    """With the solver owning the friction, B applies the motor torque and exports the budget."""
    harness = _Harness(_make_cfg(), num_envs=1, device=device)
    harness.controller.solver_applies_friction = True
    params = BamMotorParams.from_json(BAM_XL330_M6_PARAMS_FILE)

    effort = harness.step(np.array([[0.3, -0.1]]), np.array([[0.5, -0.4]]), np.zeros((1, 2)))

    np.testing.assert_allclose(effort.reshape(-1), harness.controller.motor_torque.numpy(), atol=0.0, rtol=0.0)
    budget = harness.controller.friction_budget.numpy()
    assert (budget >= params.friction_base).all(), "the published budget must keep the Coulomb floor"
    np.testing.assert_allclose(harness.controller.viscous_damping.numpy(), params.friction_viscous, atol=1e-9, rtol=0.0)


@pytest.mark.parametrize("device", test_devices())
def test_friction_scale_changes_the_opposing_torque(device):
    """Writing the ``friction_scale`` parameter must change the emitted torque.

    This is the parameter an environment's domain-randomization event drives; the write goes
    through the same controller array the group-parameter API addresses.
    """
    pos, vel, target = np.array([[0.05, 0.05]]), np.array([[0.02, 0.02]]), np.zeros((1, 2))

    baseline = _Harness(_make_cfg(), num_envs=1, device=device)
    baseline_effort = baseline.step(pos, vel, target)

    scaled = _Harness(_make_cfg(), num_envs=1, device=device)
    scaled.controller.friction_scale.fill_(4.0)
    scaled_effort = scaled.step(pos, vel, target)

    np.testing.assert_allclose(
        scaled.controller.friction_budget.numpy(),
        4.0 * baseline.controller.friction_budget.numpy(),
        rtol=1e-6,
        atol=0.0,
    )
    assert np.abs(scaled_effort - baseline_effort).max() > 1e-4


@pytest.mark.parametrize("device", test_devices())
def test_shared_supply_sags_with_the_group_load(device):
    """The supply drop is driven by the whole group's load, not by each joint's own.

    ``env_dof_stride`` is what tells the controller which flat DOFs belong to one supply; the
    adapter declares it because it is the first object that knows the environment count.
    """
    harness = _Harness(_make_cfg(vin_drop_gain_range=None), num_envs=2, device=device)
    assert harness.controller.env_dof_stride == len(JOINT_NAMES)
    harness.controller.sag_gain.fill_(5.0)

    pos = np.array([[0.4, 0.4], [0.4, 0.4]])
    target = np.zeros((2, 2))
    harness.step(pos, np.zeros((2, 2)), target)
    # ``numpy()`` aliases a Warp array on the host, so the torque has to be copied out
    # before the next step overwrites it.
    motor = harness.controller.motor_torque.numpy().copy()
    harness.step(pos, np.zeros((2, 2)), target)

    expected = VIN - 5.0 * np.abs(motor.reshape(2, 2)).sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(
        harness.controller.effective_vin.numpy().reshape(2, 2), np.broadcast_to(expected, (2, 2)), rtol=1e-5, atol=0.0
    )


@pytest.mark.parametrize("device", test_devices())
def test_startup_sampling_draws_one_value_per_environment(device):
    """The config's start-up ranges must reach the controller once the actuator exists.

    A USD prim is shared by every clone, so the ranges cannot be authored per environment.
    They are drawn afterwards, and -- like implementation A's ``_sample_per_env`` -- one value
    covers all of an environment's joints.
    """
    cfg = _make_cfg(vin_range=(6.0, 8.0), friction_scale_range=(0.5, 1.5))
    harness = _Harness(cfg, num_envs=8, device=device)

    apply_bam_startup_sampling(harness.controller, cfg)

    for attr, (low, high) in (("vin", cfg.vin_range), ("friction_scale", cfg.friction_scale_range)):
        values = getattr(harness.controller, attr).numpy().reshape(8, len(JOINT_NAMES))
        np.testing.assert_allclose(values[:, 0], values[:, 1], atol=0.0, rtol=0.0)
        assert ((values >= low) & (values <= high)).all()
        assert len(np.unique(values[:, 0])) > 1, "every environment drew the same value"
    # An unset range leaves the authored nominal in place.
    np.testing.assert_allclose(harness.controller.sag_gain.numpy(), 0.0, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("device", test_devices())
def test_constant_delay_replays_an_older_command(device):
    """A fixed lag of ``k`` steps must reproduce an undelayed actuator fed the ``k``-step-old command."""
    lag = 3
    delayed = _Harness(_make_cfg(min_delay=lag, max_delay=lag), num_envs=1, device=device)
    undelayed = _Harness(_make_cfg(), num_envs=1, device=device)

    commands = [np.full((1, 2), 0.1 * step) for step in range(8)]
    pos, vel = np.zeros((1, 2)), np.zeros((1, 2))
    for step, command in enumerate(commands):
        got = delayed.step(pos, vel, command)
        # The ring clamps to the oldest command it has seen, exactly like the reference buffer.
        expected = undelayed.step(pos, vel, commands[max(step - lag, 0)])
        np.testing.assert_allclose(got, expected, atol=1e-6, rtol=0.0, err_msg=f"step {step}")


@pytest.mark.parametrize("device", test_devices())
def test_reset_restores_the_first_step_behaviour(device):
    """Resetting an environment must clear its caches without touching the others."""
    harness = _Harness(_make_cfg(), num_envs=2, device=device)
    pos, vel, target = np.array([[0.2, 0.2], [0.2, 0.2]]), np.array([[1.0, 1.0], [1.0, 1.0]]), np.zeros((2, 2))

    first = harness.step(pos, vel, target)
    harness.step(pos, vel, target)
    harness.reset(torch.tensor([0], device=device))
    after = harness.step(pos, vel, target)

    np.testing.assert_allclose(after[0], first[0], atol=1e-6, rtol=0.0)
    assert np.abs(after[1] - first[1]).max() > 1e-9, "the untouched environment must keep its history"


"""
Encoder-through-backlash feedback.
"""


def _interleave(servo: np.ndarray, play: np.ndarray) -> np.ndarray:
    """Lay per-servo and per-play values out over :data:`BACKLASH_JOINT_NAMES`' joint order."""
    full = np.zeros((servo.shape[0], len(BACKLASH_JOINT_NAMES)))
    full[:, SERVO_SLOTS] = servo
    full[:, PLAY_SLOTS] = play
    return full


def _backlash_binding(device: str, mask: list[float], num_envs: int = 1) -> tuple[wp.array, wp.array]:
    """Build the per-DOF binding :mod:`isaaclab_newton` will resolve from the joint names.

    Returns the flat index of each driven DOF's ``passive_<joint>_backlash`` hinge in the
    position array, and one mask entry per driven DOF, both in the actuator's DOF order
    (environment major).
    """
    stride = len(BACKLASH_JOINT_NAMES)
    indices = [PLAY_SLOTS[slot] + env * stride for env in range(num_envs) for slot in range(len(SERVO_SLOTS))]
    return (
        wp.array(np.array(indices, dtype=np.uint32), device=device),
        wp.array(np.array(mask * num_envs, dtype=np.float32), device=device),
    )


def _make_bound_harness(device: str, mask: list[float], num_envs: int = 1) -> _Harness:
    """Build the serial-play fixture with each servo bound to the hinge that follows it."""
    harness = _Harness(
        _make_cfg(joint_names_expr=SERVO_ONLY_EXPR), num_envs=num_envs, device=device, joint_names=BACKLASH_JOINT_NAMES
    )
    harness.controller.bind_backlash_indices(*_backlash_binding(device, mask, num_envs))
    return harness


def _reference_encoder_efforts(measured: np.ndarray, motor_vel: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Roll the shared BAM math core forward over a scripted sequence.

    Built from :mod:`isaaclab.actuators.bam_model` rather than from either actuator, so the
    expectation is an independent calculation and not the implementation restated. The only
    backlash-specific input is *measured*, which the caller composes as ``servo + play * mask``:
    upstream rewrites the firmware command's position feedback and nothing else
    (``friction_dr_bam.py:101-104``), so the velocity that reaches the back-EMF, the Stribeck
    coefficient and the stiction clip here is the motor-side one.

    The fixture leaves the delay off, the friction scale at one and the supply sag at zero, so
    the recursion carries only the three caches the controller keeps: the previous motor torque,
    the previous applied torque and the previous velocity.

    Args:
        measured: Encoder-view positions the firmware closes its loop on [rad], shape
            ``(steps, num_envs, num_joints)``.
        motor_vel: Motor-side joint velocities [rad/s], same shape.
        target: Commanded positions [rad], same shape.

    Returns:
        Applied efforts [N.m], same shape.
    """
    from isaaclab.actuators.bam_model import (  # noqa: PLC0415
        apply_stiction_clip,
        battery_sag,
        compute_duty,
        compute_friction_budget,
        compute_motor_torque,
        compute_stribeck_coeff,
    )

    params = BamMotorParams.from_json(BAM_XL330_M6_PARAMS_FILE)
    steps, num_envs, num_joints = measured.shape
    kp = torch.full((num_envs, 1), KP_FW)
    vin = torch.full((num_envs, 1), VIN)
    prev_motor = torch.zeros(num_envs, num_joints)
    prev_applied = torch.zeros(num_envs, num_joints)
    prev_vel = None

    efforts = []
    for step in range(steps):
        q, dq, q_target = (torch.tensor(a[step], dtype=torch.float32) for a in (measured, motor_vel, target))
        # A freshly built controller seeds its velocity cache, so the first step sees no
        # acceleration and hence no estimated external torque.
        if prev_vel is None:
            prev_vel = dq
        effective_vin = battery_sag(vin, prev_motor, torch.zeros(num_envs, 1), None)
        duty = compute_duty(q_target, q, dq, kp, effective_vin, params)
        motor = compute_motor_torque(duty, dq, effective_vin, params)
        external = params.armature * (dq - prev_vel) / DT - prev_applied
        budget = compute_friction_budget(prev_motor, external, compute_stribeck_coeff(dq, params), params, 1.0)
        applied = apply_stiction_clip(motor, external, dq, budget, params.friction_viscous, DT, params.armature)
        efforts.append(applied.numpy())
        prev_motor, prev_applied, prev_vel = motor, applied, dq
    return np.stack(efforts)


@pytest.mark.parametrize("device", test_devices())
def test_bound_encoder_closes_the_firmware_loop_through_the_play(device):
    """The firmware error must be measured against ``servo + play``, not against the servo.

    On the real servo the magnetic encoder sits on the *output* side of the gear play, so while
    the rotor winds through the dead zone the position the firmware reads -- and hence its
    proportional error -- does not move. That is the whole point of the backlash plant: without
    it the play is a compliance the policy never sees, with it the policy inherits the dead
    zone. The expectation is the shared math core stepped over the same scripted sequence with
    the feedback composed the same way.

    The two joints carry different masks on purpose: one reads through its hinge, the other does
    not, and both are resolved inside the same kernel launch, so a mask applied per launch
    rather than per DOF fails here.
    """
    mask = [1.0, 0.0]
    harness = _make_bound_harness(device, mask)

    rng = np.random.default_rng(11)
    steps, shape = 12, (1, len(SERVO_SLOTS))
    servo_pos = rng.uniform(-0.4, 0.4, (steps, *shape))
    play_pos = rng.uniform(-PLAY_LIMIT, PLAY_LIMIT, (steps, *shape))
    servo_vel = rng.uniform(-2.0, 2.0, (steps, *shape))
    play_vel = rng.uniform(-2.0, 2.0, (steps, *shape))
    target = rng.uniform(-0.4, 0.4, (steps, *shape))
    zeros = np.zeros(shape)

    got = np.stack(
        [
            harness.step(
                _interleave(servo_pos[step], play_pos[step]),
                _interleave(servo_vel[step], play_vel[step]),
                _interleave(target[step], zeros),
            )[:, SERVO_SLOTS]
            for step in range(steps)
        ]
    )

    expected = _reference_encoder_efforts(servo_pos + play_pos * np.array(mask), servo_vel, target)
    np.testing.assert_allclose(got, expected, atol=1e-6, rtol=0.0)

    # One degree of play is a small angle; the comparison above only means something if reading
    # through it moves the torque by far more than the tolerance it was asserted at.
    without_play = _reference_encoder_efforts(servo_pos, servo_vel, target)
    assert np.abs(expected[..., 0] - without_play[..., 0]).max() > 1e-3


@pytest.mark.parametrize("device", test_devices())
def test_the_play_hinges_velocity_never_reaches_the_motor(device):
    """Only the position feedback reads through the play; the velocity stays motor-side.

    In this model the joint velocity drives the back-EMF, the Stribeck blend and the stopping
    torque of the stiction clip -- rotor physics, not an encoder-derived firmware signal. The
    reference implementation is explicit about leaving it alone
    (``friction_dr_bam.py:78-80``), and summing the hinge in there would damp the motor against
    a velocity its rotor never sees.
    """
    baseline = _make_bound_harness(device, [1.0, 1.0])
    disturbed = _make_bound_harness(device, [1.0, 1.0])

    rng = np.random.default_rng(5)
    for step in range(6):
        servo_pos = rng.uniform(-0.4, 0.4, (1, 2))
        play_pos = rng.uniform(-PLAY_LIMIT, PLAY_LIMIT, (1, 2))
        servo_vel = rng.uniform(-2.0, 2.0, (1, 2))
        target = rng.uniform(-0.4, 0.4, (1, 2))
        quiet = baseline.step(
            _interleave(servo_pos, play_pos),
            _interleave(servo_vel, np.zeros((1, 2))),
            _interleave(target, np.zeros((1, 2))),
        )
        spinning = disturbed.step(
            _interleave(servo_pos, play_pos),
            _interleave(servo_vel, rng.uniform(-20.0, 20.0, (1, 2))),
            _interleave(target, np.zeros((1, 2))),
        )
        np.testing.assert_array_equal(
            spinning[:, SERVO_SLOTS], quiet[:, SERVO_SLOTS], err_msg=f"the hinge velocity leaked at step {step}"
        )


@pytest.mark.parametrize("device", test_devices())
def test_a_zero_mask_reproduces_the_plain_controller_bit_for_bit(device):
    """A joint with no play must reach exactly the torque it reached before the encoder view.

    One configuration has to be safe on every model -- that is what makes the mask worth having
    rather than a second controller class -- so a plant without play hinges may not cost
    anything at all. Not "almost nothing": the sum the kernel now evaluates has to collapse to
    the old expression exactly, or every policy trained against the plain asset faces a
    different plant. The reference is the plain fixture, whose articulation has no play hinges
    in it; the serial-play fixtures hold nonzero hinge positions throughout, so a mask that
    leaked would show up immediately.
    """
    plain = _Harness(_make_cfg(), num_envs=1, device=device)
    unbound = _Harness(
        _make_cfg(joint_names_expr=SERVO_ONLY_EXPR), num_envs=1, device=device, joint_names=BACKLASH_JOINT_NAMES
    )
    zero_masked = _make_bound_harness(device, [0.0, 0.0])
    engaged = _make_bound_harness(device, [1.0, 1.0])

    rng = np.random.default_rng(3)
    engaged_gap = 0.0
    for step in range(8):
        servo_pos = rng.uniform(-0.4, 0.4, (1, 2))
        play_pos = rng.uniform(-PLAY_LIMIT, PLAY_LIMIT, (1, 2))
        servo_vel = rng.uniform(-2.0, 2.0, (1, 2))
        target = rng.uniform(-0.4, 0.4, (1, 2))
        pos, vel, cmd = (
            _interleave(servo_pos, play_pos),
            _interleave(servo_vel, np.zeros((1, 2))),
            _interleave(target, np.zeros((1, 2))),
        )

        reference = plain.step(servo_pos, servo_vel, target)
        for name, harness in (("unbound", unbound), ("zero-masked", zero_masked)):
            got = harness.step(pos, vel, cmd)[:, SERVO_SLOTS]
            np.testing.assert_array_equal(got, reference, err_msg=f"the {name} controller diverged at step {step}")
        engaged_gap = max(engaged_gap, np.abs(engaged.step(pos, vel, cmd)[:, SERVO_SLOTS] - reference).max())

    # The mutation check the exactness above needs: the same fixture with the mask raised has to
    # leave the plain controller's trajectory, or both halves of this test are asserting nothing.
    assert engaged_gap > 1e-3, "raising the mask left the torque unchanged"


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_the_bound_encoder_view_replays_from_a_cuda_graph(device):
    """The encoder feedback must run inside a captured graph, reading the live binding.

    A controller that forced a host round trip would cost the whole decimation loop its capture,
    so the property is asserted rather than assumed. The capture holds an even number of steps
    on purpose: the actuator state is double-buffered and swapped in Python, so an odd capture
    drops the last update on every replay -- the same reason the environment-level harness
    captures an even decimation. Rebinding between replays is what shows the indices are read
    from the controller's arrays on every launch rather than baked into the graph.
    """
    servo_pos, play_pos = np.array([[0.21, -0.13]]), np.array([[PLAY_LIMIT, -PLAY_LIMIT]])
    pos = _interleave(servo_pos, play_pos)
    vel = _interleave(np.array([[0.7, -0.9]]), np.zeros((1, 2)))
    cmd = _interleave(np.array([[0.05, 0.05]]), np.zeros((1, 2)))
    all_envs = torch.zeros(1, dtype=torch.long, device=device)

    eager = _make_bound_harness(device, [1.0, 1.0])
    eager.step(pos, vel, cmd)
    expected = eager.step(pos, vel, cmd)[:, SERVO_SLOTS].copy()

    captured = _make_bound_harness(device, [1.0, 1.0])
    # Warp modules have to be resident before a capture; one eager step loads them, and the
    # reset that follows puts the controller back to its first-step behaviour.
    captured.step(pos, vel, cmd)
    captured.reset(all_envs)
    with wp.ScopedDevice(device), wp.ScopedCapture() as capture:
        for _ in range(2):
            captured.adapter.step(captured.state, captured.control, DT)

    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    np.testing.assert_allclose(captured.control.joint_f_2d.numpy()[:, SERVO_SLOTS], expected, atol=1e-6, rtol=0.0)

    captured.controller.bind_backlash_indices(*_backlash_binding(device, [0.0, 0.0]))
    captured.reset(all_envs)
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    assert np.abs(captured.control.joint_f_2d.numpy()[:, SERVO_SLOTS] - expected).max() > 1e-3, (
        "the replayed graph kept the old mask, so the binding was baked in at capture"
    )
