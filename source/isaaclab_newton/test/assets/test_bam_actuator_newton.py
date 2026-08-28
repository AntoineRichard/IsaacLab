# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end checks of the Lab-executed BAM actuator on a live Newton MJWarp articulation.

:class:`~isaaclab.actuators.BamActuator` is an explicit model: with
``SimulationCfg.use_newton_actuators=False`` its efforts are computed in Python and written
to the solver as a pure joint-effort command. The unit suite
(``source/isaaclab/test/actuators/test_bam_actuator.py``) pins the model against the math
core with hand-fed positions and velocities; these tests close the loop by letting MJWarp
integrate the efforts and by checking that the resting state of a real pendulum is the one
the model predicts.

The asset is a fixed-base single-degree-of-freedom pendulum authored by the test itself, so
the load is a closed-form ``m * g * L * cos(theta)`` and the analytic equilibrium follows
from the math core alone. It is written to a temporary file rather than checked in: the
repository does not track USD assets.

The nearest existing sibling, ``test_newton_actuators_newton.py``, constructs an
:class:`~isaaclab.app.AppLauncher` at import time and therefore cannot run without an Isaac
Sim runtime. This file follows the Kit-less pattern of the Newton sensor suites instead
(``test/sensors/test_joint_wrench_sensor.py``): no ``AppLauncher`` and a
:func:`~isaaclab.sim.build_simulation_context` fixture. Nothing is fetched from Nucleus.
"""

import math

import numpy as np
import pytest
import torch
from isaaclab_newton.physics import (
    FeatherstoneSolverCfg,
    MjWarpActuatorBridge,
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonManager,
)

import isaaclab.sim as sim_utils
from isaaclab.actuators import BamActuator, BamActuatorCfg
from isaaclab.actuators.bam_model import (
    BamMotorParams,
    compute_duty,
    compute_friction_budget,
    compute_motor_torque,
)
from isaaclab.actuators.newton import ControllerBam, read_group_parameter, write_group_parameter
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.test.utils import test_devices

pytestmark = [pytest.mark.integration, pytest.mark.kitless]

PENDULUM_USDA = """\
#usda 1.0
(
    defaultPrim = "Robot"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Robot" (
    prepend apiSchemas = ["PhysicsArticulationRootAPI"]
)
{
    def Xform "Pivot" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {
        float physics:mass = 0.1
        float3 physics:diagonalInertia = (0.001, 0.001, 0.001)
    }

    def Xform "Arm" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {
        float physics:mass = 0.02
        point3f physics:centerOfMass = (0.05, 0, 0)
        float3 physics:diagonalInertia = (0.002, 0.002, 0.002)
    }

    def PhysicsFixedJoint "anchor"
    {
        rel physics:body1 = </Robot/Pivot>
    }

    def PhysicsRevoluteJoint "joint"
    {
        uniform token physics:axis = "Y"
        rel physics:body0 = </Robot/Pivot>
        rel physics:body1 = </Robot/Arm>
    }
}
"""
"""Fixed-base single-degree-of-freedom pendulum.

``Pivot`` is welded to the world by ``anchor``, a fixed joint whose ``physics:body0`` is left
unset. That weld is what makes the USD physics parser report an articulation at all: a lone
body hanging off a world-anchored revolute joint is imported as an orphan joint and no
articulation is created, so :class:`~isaaclab.assets.Articulation` finds nothing to bind to.
``Arm`` is the only degree of freedom.

``Arm``'s centre of mass is offset along +X and the joint spins about +Y, so rotating the
joint by ``theta`` swings the centre of mass down and gravity applies a pure
``m * g * L * cos(theta)`` load about the joint axis.

``Arm``'s inertia is authored independently of its mass distribution and is deliberately much
larger than ``m * L^2``: it stands in for the rotor inertia a geared servo reflects to its
output shaft (0.0018 kg m^2 for the Dynamixel XL330 the BAM parameters were fitted on).
Without it the joint inertia would sit far below the actuator's electrical damping times the
timestep, and a back-EMF torque applied explicitly by the actuator could not be integrated
stably.
"""

DT = 1.0 / 120.0
"""Physics timestep [s]."""

NUM_ENVS = 2
"""Environments simulated side by side, so a per-environment randomization is observable."""

NUM_STEPS = 200
"""Steps each settling phase runs for [-]. The joint is at rest well before this."""

VIN = 7.4
"""Supply voltage the actuator is configured with [V]."""

KP_FW = 200.0
"""Firmware proportional gain the actuator is configured with [-]."""

INITIAL_ANGLE = 0.3
"""Angle the arm is released from [rad]. The commanded target is always 0."""

ANGLE_TOLERANCE = math.radians(2.0)
"""Accepted distance between the settled angle and the analytic equilibrium [rad]."""


def _make_sim_cfg(device: str, use_newton_actuators: bool = False, use_cuda_graph: bool = True) -> SimulationCfg:
    """Build the MJWarp configuration used by every test in this module."""
    return SimulationCfg(
        dt=DT,
        device=device,
        use_newton_actuators=use_newton_actuators,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(njmax=20, nconmax=20, ls_iterations=20, integrator="implicitfast", impratio=1),
            num_substeps=2,
            debug_mode=False,
            use_cuda_graph=use_cuda_graph,
        ),
    )


@pytest.fixture(scope="module")
def pendulum_usd(tmp_path_factory) -> str:
    """Write :data:`PENDULUM_USDA` to a temporary file and return its path."""
    path = tmp_path_factory.mktemp("bam_pendulum") / "single_joint_pendulum.usda"
    path.write_text(PENDULUM_USDA)
    return str(path)


@pytest.fixture
def sim(device):
    """Newton simulation context running the Isaac Lab-executed actuator path."""
    with build_simulation_context(
        device=device,
        gravity_enabled=True,
        add_ground_plane=False,
        sim_cfg=_make_sim_cfg(device),
    ) as sim_ctx:
        sim_ctx._app_control_on_stop_handle = None  # noqa: SLF001
        yield sim_ctx


@pytest.fixture
def native_sim(device):
    """Newton simulation context running the Newton-native actuator path."""
    with build_simulation_context(
        device=device,
        gravity_enabled=True,
        add_ground_plane=False,
        sim_cfg=_make_sim_cfg(device, use_newton_actuators=True),
    ) as sim_ctx:
        sim_ctx._app_control_on_stop_handle = None  # noqa: SLF001
        yield sim_ctx


@pytest.fixture
def native_sim_eager(device):
    """Newton-native actuator path with CUDA graph capture off.

    A replayed graph runs its recorded kernels without re-entering Python, so a test that
    observes the hooks from Python has to step eagerly to see every iteration on both devices.
    """
    with build_simulation_context(
        device=device,
        gravity_enabled=True,
        add_ground_plane=False,
        sim_cfg=_make_sim_cfg(device, use_newton_actuators=True, use_cuda_graph=False),
    ) as sim_ctx:
        sim_ctx._app_control_on_stop_handle = None  # noqa: SLF001
        yield sim_ctx


def _build_pendulum(sim, pendulum_usd: str) -> Articulation:
    """Spawn :data:`NUM_ENVS` BAM-driven pendulums and initialize the simulation."""
    for index in range(NUM_ENVS):
        sim_utils.create_prim(f"/World/Env_{index}", "Xform", translation=(index * 1.0, 0.0, 1.0))
    robot = Articulation(
        ArticulationCfg(
            prim_path="/World/Env_[^/]*/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path=pendulum_usd),
            init_state=ArticulationCfg.InitialStateCfg(joint_pos={"joint": INITIAL_ANGLE}),
            actuators={"servo": BamActuatorCfg(joint_names_expr=[".*"], vin=VIN, kp_fw=KP_FW)},
        )
    )
    sim.reset()
    assert robot.is_initialized
    return robot


def _gravity_load(robot: Articulation, sim) -> float:
    """Return the peak gravity torque ``m * g * L`` of the arm [N.m], read from the sim.

    Deriving the load from the live articulation rather than from the authored numbers keeps
    the reference calculation honest about what the importer actually built (a stage-unit
    misreading, for instance, would show up here rather than silently shifting the
    prediction).
    """
    arm = robot.body_names.index("Arm")
    mass = float(robot.data.body_mass.torch[0, arm])
    lever = float(robot.data.body_com_pos_b.torch[0, arm, 0])
    return mass * abs(sim.cfg.gravity[2]) * lever


def _bam_actuator_groups(robot: Articulation) -> dict[str, BamActuator]:
    """Discover the BAM groups of an articulation the way an event term would.

    :class:`~isaaclab.envs.mdp.randomize_actuator_gains` resolves its targets by iterating
    ``asset.actuators`` -- an :class:`~isaaclab.actuators.ActuatorCollection`, which is a
    public ``Mapping`` -- branching on the actuator kind, and then writing per-environment
    rows on the live instances. A ``BamActuator`` is neither implicit, nor Newton-native, nor
    an ``IdealPDActuator``, so that term skips it; a BAM randomization term performs the same
    discovery and drives the model's own hooks instead. No helper on the actuator module is
    needed for that: the collection's mapping interface is enough.
    """
    return {name: actuator for name, actuator in robot.actuators.items() if isinstance(actuator, BamActuator)}


"""
Reference calculation: the resting states the BAM math core predicts for this pendulum.
"""


def _static_terms(
    angles: torch.Tensor, load: float, *, kp_scale: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(motor torque, gravity torque, friction budget)`` of a joint held at rest.

    All three come from :mod:`isaaclab.actuators.bam_model`, evaluated at zero velocity:

    * the firmware controller and the DC-motor equation give the motor-side torque of a joint
      standing at ``angles`` with a target of zero;
    * the pendulum's load about the joint axis is ``m * g * L * cos(theta)``, because the arm's
      centre of mass sits at ``(L, 0, 0)`` and the joint spins about ``+Y``;
    * the friction budget is the one :meth:`~isaaclab.actuators.BamActuator.compute` sizes on
      a resting joint. Its ``prev_tau`` is the previous step's motor torque, which at rest is
      the motor torque above, and its ``ext_tau`` is the external load the model estimates.
      At rest the estimator returns ``-tau_applied_prev``; a joint that is genuinely holding
      applies exactly ``-tau_gravity`` (see :func:`_analytic_rest`), so the estimate is the
      gravity torque. The Stribeck coefficient is 1 at zero velocity.

    Args:
        angles: Joint angles to evaluate [rad], shape ``(N,)``.
        load: Peak gravity torque ``m * g * L`` [N.m].
        kp_scale: Firmware gain multiplier the actuator is running with [-].

    Returns:
        Motor torque [N.m], gravity torque [N.m] and friction budget [N.m], each ``(N, 1)``.
    """
    params = BamMotorParams.from_json(BamActuatorCfg().params_file)
    angles = angles.reshape(-1, 1).to(torch.float64)
    zeros = torch.zeros_like(angles)
    duty = compute_duty(
        zeros,  # commanded target
        angles,
        zeros,  # at rest
        torch.full_like(angles, KP_FW * kp_scale),
        torch.full_like(angles, VIN),
        params,
    )
    motor_tau = compute_motor_torque(duty, zeros, torch.full_like(angles, VIN), params)
    gravity_tau = load * torch.cos(angles)
    budget = compute_friction_budget(motor_tau, gravity_tau, torch.ones_like(angles), params, 1.0)
    return motor_tau, gravity_tau, budget


def _analytic_rest(load: float, *, kp_scale: float = 1.0, friction_scale: float = 1.0) -> tuple[float, float, float]:
    """Return ``(equilibrium, band lower edge, band upper edge)`` of the resting joint [rad].

    Two different quantities, both derived from :func:`_static_terms`:

    * the **equilibrium** is the frictionless fixed point, the angle where the motor torque
      exactly cancels the gravity load, ``tau_motor(theta) + tau_gravity(theta) = 0``. It is
      the "PD + gravity" resting angle the joint would take with an ideal gearbox.
    * the **stiction band** is the set of angles the joint can actually be held at. BAM's
      static-friction clip returns ``-tau_ext`` -- it cancels the load outright -- whenever
      the net torque fits inside the friction budget, so every angle with
      ``|tau_motor + tau_gravity| <= budget`` is a resting state. The band is wider than a
      symmetric interval around the equilibrium because the budget is load dependent: it
      grows with the motor torque, which itself grows with the distance to the target.

    The joint is released above the equilibrium and swings down onto it, so it comes to rest
    at the first resting state it reaches, near the band's upper edge.

    Args:
        load: Peak gravity torque ``m * g * L`` [N.m].
        kp_scale: Firmware gain multiplier the actuator is running with [-].
        friction_scale: Friction-budget multiplier the actuator is running with [-].

    Returns:
        The equilibrium angle and the two edges of the stiction band [rad].
    """
    # Brackets the release angle with room to spare, at 1e-5 rad resolution -- two orders of
    # magnitude finer than the acceptance tolerance.
    angles = torch.linspace(-0.3, 0.6, 90001, dtype=torch.float64)
    motor_tau, gravity_tau, budget = _static_terms(angles, load, kp_scale=kp_scale)
    net_tau = (motor_tau + gravity_tau).squeeze(-1)
    holds = net_tau.abs() <= (budget * friction_scale).squeeze(-1)

    equilibrium = angles[net_tau.abs().argmin()]
    inside = holds.nonzero().flatten()
    assert len(inside) > 0, "the pendulum has no resting state; check the fixture's load"
    # The band must be a single interval for "the first resting state reached" to be its edge.
    assert holds[inside[0] : inside[-1] + 1].all(), "the stiction band is not a single interval"
    return float(equilibrium), float(angles[inside[0]]), float(angles[inside[-1]])


"""
Rollout helpers.
"""


def _release(robot: Articulation) -> None:
    """Put every arm back at :data:`INITIAL_ANGLE` at rest and clear the actuator state."""
    robot.write_joint_position_to_sim_index(position=torch.full_like(robot.data.joint_pos.torch, INITIAL_ANGLE))
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(robot.data.joint_vel.torch))
    robot.actuators.reset()


def _settle(robot: Articulation, sim) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Command a zero position target for :data:`NUM_STEPS` steps and record the rollout.

    Returns:
        Joint positions [rad], joint velocities [rad/s] and applied efforts [N.m], each of
        shape ``(NUM_STEPS, NUM_ENVS, 1)``.
    """
    robot.actuators.target_command.set_position_index(
        value=torch.zeros(NUM_ENVS, robot.num_joints, device=robot.device)
    )
    positions, velocities, efforts = [], [], []
    for _ in range(NUM_STEPS):
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())
        positions.append(robot.data.joint_pos.torch.clone())
        velocities.append(robot.data.joint_vel.torch.clone())
        efforts.append(robot.actuators.applied_effort.torch.clone())
    return torch.stack(positions), torch.stack(velocities), torch.stack(efforts)


def _assert_rest(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    efforts: torch.Tensor,
    velocity_tolerance: float = 1e-3,
) -> torch.Tensor:
    """Assert the rollout is finite and has come to rest, and return the final angles.

    Args:
        positions: Recorded joint positions [rad].
        velocities: Recorded joint velocities [rad/s].
        efforts: Recorded applied efforts [N.m].
        velocity_tolerance: Largest final speed that still counts as at rest [rad/s]. The
            Newton-native path needs a looser one than the Isaac Lab-executed path; see
            :data:`NATIVE_REST_TOLERANCE`.

    Returns:
        The final joint angle of each environment [rad].
    """
    for name, trace in (("position", positions), ("velocity", velocities), ("effort", efforts)):
        assert torch.isfinite(trace).all(), f"non-finite joint {name} in the rollout"
    assert velocities[-1].abs().max() < velocity_tolerance, "the pendulum has not come to rest"
    return positions[-1].reshape(NUM_ENVS)


"""
Tests.
"""


@pytest.mark.parametrize("device", test_devices())
def test_pendulum_settles_at_the_analytic_equilibrium(sim, device, pendulum_usd):
    """Settle a BAM-driven pendulum on Newton and compare it against the math core.

    This is the end-to-end proof that the Lab-executed model runs on a live MJWarp
    articulation: the efforts it returns are integrated by the solver, and the state they
    integrate to is the one the model's own equations predict.
    """
    robot = _build_pendulum(sim, pendulum_usd)
    load = _gravity_load(robot, sim)

    _release(robot)
    final_angle = _assert_rest(*_settle(robot, sim))

    equilibrium, band_low, band_high = _analytic_rest(load)
    for env in range(NUM_ENVS):
        angle = float(final_angle[env])
        assert abs(angle - equilibrium) <= ANGLE_TOLERANCE
        assert band_low <= angle <= band_high

    # The static contract of the stiction clip: a held joint applies exactly minus the load.
    holding_effort = robot.actuators.applied_effort.torch.reshape(NUM_ENVS)
    expected_effort = -load * torch.cos(final_angle)
    torch.testing.assert_close(holding_effort, expected_effort.to(holding_effort.dtype), atol=1e-5, rtol=0.0)


@pytest.mark.parametrize("device", test_devices())
def test_friction_randomization_changes_the_hanging_error(sim, device, pendulum_usd):
    """Drive :meth:`BamActuator.set_friction_scale` per environment on the live actuator.

    Friction decides how far short of the target the arm gives up, so a heavily randomized
    environment must hang further out than a lightly randomized one. Restoring the scales
    with :meth:`BamActuator.reset_friction_scale` must bring both back onto the unrandomized
    resting state.
    """
    robot = _build_pendulum(sim, pendulum_usd)
    load = _gravity_load(robot, sim)
    actuator = _bam_actuator_groups(robot)["servo"]
    env_ids = torch.arange(NUM_ENVS, device=robot.device)
    scales = (0.5, 2.0)

    actuator.set_friction_scale(env_ids, torch.tensor([[scales[0]], [scales[1]]], device=robot.device))
    _release(robot)
    randomized_angle = _assert_rest(*_settle(robot, sim))

    # More friction, more hanging error: the arm is released above the target and stops
    # earlier the wider its stiction band is.
    assert float(randomized_angle[1]) > float(randomized_angle[0])
    for env, scale in enumerate(scales):
        _, band_low, band_high = _analytic_rest(load, friction_scale=scale)
        assert band_low <= float(randomized_angle[env]) <= band_high

    actuator.reset_friction_scale(env_ids)
    torch.testing.assert_close(actuator.friction_scale, torch.ones_like(actuator.friction_scale))
    _release(robot)
    restored_angle = _assert_rest(*_settle(robot, sim))
    equilibrium, band_low, band_high = _analytic_rest(load)
    for env in range(NUM_ENVS):
        assert abs(float(restored_angle[env]) - equilibrium) <= ANGLE_TOLERANCE
        assert band_low <= float(restored_angle[env]) <= band_high


@pytest.mark.parametrize("device", test_devices())
def test_gain_randomization_changes_the_hanging_error(sim, device, pendulum_usd):
    """Drive :meth:`BamActuator.set_gains` per environment on the live actuator.

    A stiffer firmware gain both moves the equilibrium closer to the target and narrows the
    stiction band, so the randomized environment must settle nearer the target.
    :meth:`BamActuator.reset_gains` must restore both scales and the resting state.
    """
    robot = _build_pendulum(sim, pendulum_usd)
    load = _gravity_load(robot, sim)
    actuator = _bam_actuator_groups(robot)["servo"]
    env_ids = torch.arange(NUM_ENVS, device=robot.device)
    kp_scales = (1.0, 3.0)

    actuator.set_gains(env_ids, kp_scale=torch.tensor([[kp_scales[0]], [kp_scales[1]]], device=robot.device))
    _release(robot)
    randomized_angle = _assert_rest(*_settle(robot, sim))

    assert float(randomized_angle[1]) < float(randomized_angle[0])
    for env, kp_scale in enumerate(kp_scales):
        _, band_low, band_high = _analytic_rest(load, kp_scale=kp_scale)
        assert band_low <= float(randomized_angle[env]) <= band_high

    actuator.reset_gains(env_ids)
    torch.testing.assert_close(actuator.kp_scale, torch.ones_like(actuator.kp_scale))
    torch.testing.assert_close(actuator.kd_scale, torch.ones_like(actuator.kd_scale))
    _release(robot)
    restored_angle = _assert_rest(*_settle(robot, sim))
    equilibrium, band_low, band_high = _analytic_rest(load)
    for env in range(NUM_ENVS):
        assert abs(float(restored_angle[env]) - equilibrium) <= ANGLE_TOLERANCE
        assert band_low <= float(restored_angle[env]) <= band_high


"""
Newton-native path (implementation B).

The same configuration, with ``use_newton_actuators=True``, runs the BAM pipeline as Warp
kernels inside the actuator fast path and publishes its friction budget into MuJoCo's
``dof_frictionloss`` instead of clipping the torque itself. Two contracts change with it and
are asserted below:

* the *held effort* is the bare motor torque, not ``-tau_gravity``. The load is cancelled by
  the solver's friction-loss constraint, which does not show up in the actuator's telemetry;
* the resting states are still the math core's stiction band. At rest the true external torque
  the bridge reads (``-qfrc_bias + qfrc_constraint`` minus the actuator's own friction rows)
  equals the gravity load, which is exactly what implementation A's estimator converges to,
  and the budget is sized from the same previous motor torque. The band is therefore the same
  interval :func:`_analytic_rest` returns; what differs is only *how* the joint is held inside
  it -- by a constraint whose softness is countered by
  :attr:`~isaaclab.actuators.BamActuatorCfg.stiff_frictionloss`.
"""

NATIVE_REST_TOLERANCE = 5e-3
"""Largest final speed the Newton-native path counts as at rest [rad/s].

MuJoCo's friction-loss constraint is compliant, not a hard stop: even with the stiffened
solver reference the reference implementation uses, a held joint keeps creeping at order
1e-3 rad/s instead of stopping dead the way the torque-level clip of implementation A does.
That is three orders of magnitude below the 0.6 rad/s the arm is released with, and the
residual drift is bounded separately by :func:`_assert_creep_is_bounded`. Tightening this
threshold would not measure a better actuator, only a stiffer constraint.
"""

NATIVE_MAX_CREEP = math.radians(0.2)
"""Largest angle the native path may drift over the last :data:`NATIVE_CREEP_WINDOW` steps [rad]."""

NATIVE_CREEP_WINDOW = 50
"""Trailing window the creep bound is measured over [physics steps]."""

GRAPH_DECIMATION = 2
"""Decimation the graph-capture test runs at [physics steps per environment step].

Even by necessity, not by taste: see the test's docstring.
"""


NATIVE_FRICTION_SEPARATION = math.radians(2.0)
"""Smallest hanging-error gap the 0.5 / 2.0 friction scales must open up [rad].

The scales are 4x apart, which measures as roughly 4 degrees on this fixture. Requiring a
real separation, not just an ordering, is what makes the assertion fail if the budget never
reaches the solver: two environments running the same published friction settle together and
their order is then decided by rounding.
"""

FRICTION_SCALES = (0.5, 2.0)
"""Per-environment friction-budget scales the randomization tests write [-]."""


def _assert_creep_is_bounded(positions: torch.Tensor) -> None:
    """Assert the settled joint is not sliding away under its own friction constraint."""
    drift = (positions[-1] - positions[-1 - NATIVE_CREEP_WINDOW]).abs().max()
    assert float(drift) < NATIVE_MAX_CREEP, f"the held joint drifted {float(drift):.2e} rad while nominally at rest"


def _settle_with_friction_scales(robot: Articulation, sim, load: float) -> torch.Tensor:
    """Write :data:`FRICTION_SCALES` per environment, settle, and assert the effect.

    The write goes through the group-parameter API -- the path an environment's
    domain-randomization event uses -- and lands in the controller array the in-graph friction
    publish reads, so this exercises the whole chain from the event down to MuJoCo's
    constraint.

    Returns:
        The settled joint angle of each environment [rad].
    """
    expected = torch.tensor([[FRICTION_SCALES[0]], [FRICTION_SCALES[1]]], device=robot.device)
    write_group_parameter(robot.actuators, "servo", "controller", "friction_scale", expected)
    torch.testing.assert_close(read_group_parameter(robot.actuators, "servo", "controller", "friction_scale"), expected)

    _release(robot)
    rollout = _settle(robot, sim)
    settled = _assert_rest(*rollout, velocity_tolerance=NATIVE_REST_TOLERANCE)
    _assert_creep_is_bounded(rollout[0])

    # More friction, more hanging error: the arm is released above the target and stops
    # earlier the wider its stiction band is.
    separation = float(settled[1]) - float(settled[0])
    assert separation > NATIVE_FRICTION_SEPARATION, f"the friction scales barely separated ({separation:.2e} rad)"
    for env, scale in enumerate(FRICTION_SCALES):
        _, band_low, band_high = _analytic_rest(load, friction_scale=scale)
        assert band_low <= float(settled[env]) <= band_high
    return settled


def _build_native_pendulum(sim, pendulum_usd: str) -> Articulation:
    """Spawn :data:`NUM_ENVS` BAM-driven pendulums on the Newton-native actuator path."""
    for index in range(NUM_ENVS):
        sim_utils.create_prim(f"/World/Env_{index}", "Xform", translation=(index * 1.0, 0.0, 1.0))
    robot = Articulation(
        ArticulationCfg(
            prim_path="/World/Env_[^/]*/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path=pendulum_usd),
            init_state=ArticulationCfg.InitialStateCfg(joint_pos={"joint": INITIAL_ANGLE}),
            actuators={"servo": BamActuatorCfg(joint_names_expr=[".*"], vin=VIN, kp_fw=KP_FW)},
        )
    )
    sim.reset()
    assert robot.is_initialized
    assert "servo" in robot.actuators._native_group_names, "the BAM group must run on the Newton path"
    return robot


def _native_controller(robot: Articulation) -> ControllerBam:
    """Return the BAM controller the articulation's native group is executed by."""
    controllers = [
        actuator.controller
        for actuator in NewtonManager._adapter.actuators
        if isinstance(actuator.controller, ControllerBam)
    ]
    assert len(controllers) == 1, "the fixture has exactly one BAM actuator"
    return controllers[0]


@pytest.mark.parametrize("device", test_devices())
def test_native_pendulum_settles_inside_the_stiction_band(native_sim, device, pendulum_usd):
    """Settle a Newton-native BAM pendulum and check it against the math core.

    This is the end-to-end proof of implementation B: the Warp controller's motor torque and
    the friction budget it publishes are integrated by MuJoCo, and the state they integrate to
    is the one the shared math core predicts.
    """
    robot = _build_native_pendulum(native_sim, pendulum_usd)
    load = _gravity_load(robot, native_sim)
    controller = _native_controller(robot)
    assert controller.solver_applies_friction, "the MuJoCo solver must own the friction budget"
    # Authoring seeds a positive joint friction on the driven joints. MuJoCo only assembles a
    # friction-loss constraint row where the frictionloss is positive, and it sizes its
    # constraint budget from the model as spawned, so the row has to exist before the first
    # solve; the per-step budget then overwrites the value.
    assert (NewtonManager._model.joint_friction.numpy() > 0.0).all()

    _release(robot)
    rollout = _settle(robot, native_sim)
    final_angle = _assert_rest(*rollout, velocity_tolerance=NATIVE_REST_TOLERANCE)
    _assert_creep_is_bounded(rollout[0])

    _, band_low, band_high = _analytic_rest(load)
    for env in range(NUM_ENVS):
        assert band_low <= float(final_angle[env]) <= band_high

    # The solver-side contract: the actuator applies the motor torque and nothing else; the
    # load is cancelled by the friction-loss constraint, which is invisible to this telemetry.
    holding_effort = robot.actuators.applied_effort.torch.reshape(NUM_ENVS)
    motor_torque = torch.as_tensor(controller.motor_torque.numpy(), device=holding_effort.device)
    torch.testing.assert_close(holding_effort, motor_torque.reshape(NUM_ENVS), atol=1e-6, rtol=0.0)
    # ... and it is a real torque, not a dead actuator sitting at zero.
    assert holding_effort.abs().min() > 1e-3

    # The published budget is what MuJoCo is clipping with.
    frictionloss = NewtonManager._solver.mjw_model.dof_frictionloss.numpy().reshape(-1)
    np.testing.assert_allclose(frictionloss, controller.friction_budget.numpy(), atol=1e-7, rtol=0.0)

    # The load the gearbox works against is read from the solver rather than estimated, and at
    # rest that read is exact: it is the gravity torque to the last digit. It also proves the
    # actuator's own friction rows are stripped out of the constraint force -- leaving them in
    # would report roughly twice this value here and feed the friction back on itself.
    external_torque = torch.as_tensor(controller.external_torque.numpy(), device=final_angle.device)
    expected_load = (load * torch.cos(final_angle)).to(external_torque.dtype)
    torch.testing.assert_close(external_torque.reshape(NUM_ENVS), expected_load, atol=1e-6, rtol=0.0)


@pytest.mark.parametrize("device", test_devices())
def test_native_friction_randomization_changes_the_hanging_error(native_sim, device, pendulum_usd):
    """Randomize ``friction_scale`` per environment through the group-parameter API.

    This is the write path an environment's domain-randomization event uses, and it addresses
    the controller by the same attribute name :class:`~isaaclab.actuators.BamActuator` exposes,
    so one event term drives both implementations.
    """
    robot = _build_native_pendulum(native_sim, pendulum_usd)

    # All five randomizable quantities are addressable under the names
    # :class:`~isaaclab.actuators.BamActuator` uses, so a single event term covers both paths.
    for attr in ("vin", "sag_gain", "friction_scale", "kp_scale", "kd_scale"):
        values = read_group_parameter(robot.actuators, "servo", "controller", attr)
        assert values.shape == (NUM_ENVS, robot.num_joints)

    _settle_with_friction_scales(robot, native_sim, _gravity_load(robot, native_sim))


@pytest.mark.parametrize("device", test_devices())
def test_native_bam_actuators_are_captured_in_the_cuda_graph(native_sim, device, pendulum_usd):
    """The BAM controller and its friction publish must run inside the captured graph.

    A component that forced eager stepping would cost the whole decimation loop its capture,
    so the property is asserted rather than assumed, and the settling is then re-checked
    through the replayed graph. Capture happens on
    :meth:`~isaaclab_newton.physics.NewtonManager.set_decimation`, which an environment calls
    for its policy decimation; a bare simulation context never does, so the test calls it.

    The decimation is even on purpose. A captured loop with an odd number of actuator steps
    drops the last update of the double-buffered actuator state on every replay -- a Newton
    backend property that predates this actuator and that
    ``NewtonManager._check_actuator_state_capture_balance`` reports.
    """
    robot = _build_native_pendulum(native_sim, pendulum_usd)
    load = _gravity_load(robot, native_sim)
    assert NewtonManager._adapter.is_all_graphable
    assert NewtonManager._is_all_graphable()
    assert NewtonManager._pre_actuator_callbacks, "the external-torque gather must be registered"

    NewtonManager.set_decimation(GRAPH_DECIMATION)
    if device.startswith("cuda"):
        assert NewtonManager._graph is not None, "the decimation loop was not captured"

    # A graph that baked a stale friction budget would settle both environments together, so
    # replaying it under a per-environment randomization is the real capture evidence: the
    # in-graph publish has to read the controller array a host-side write just changed.
    _settle_with_friction_scales(robot, native_sim, load)


@pytest.mark.parametrize("device", test_devices())
def test_the_friction_budget_is_refreshed_on_every_physics_step(native_sim_eager, device, pendulum_usd):
    """Load gather, actuator step and friction publish must run once per *physics* step.

    The BAM friction budget is sized from the previous solve's generalized load, so a budget
    computed once per *control* step would face a load up to ``decimation`` solves old. The
    one-step lag is deliberate -- it is what the reference implementation carries -- but a
    ``decimation``-step lag is not, and nothing else in this suite distinguishes the two: both
    leave the same value in ``dof_frictionloss`` when the step returns.

    The order within an iteration matters just as much and is asserted with the cadence: the
    gather has to read the previous solve before the actuators run, and the publish has to land
    after them and before the substeps consume the row.
    """
    robot = _build_native_pendulum(native_sim_eager, pendulum_usd)
    events: list[str] = []
    gather, publish = MjWarpActuatorBridge.gather_external_torque, MjWarpActuatorBridge.publish_dof_friction
    step = NewtonManager._adapter.step

    def spy(name, wrapped):
        def wrapper(*args, **kwargs):
            events.append(name)
            return wrapped(*args, **kwargs)

        return wrapper

    MjWarpActuatorBridge.gather_external_torque = spy("gather", gather)
    MjWarpActuatorBridge.publish_dof_friction = spy("publish", publish)
    NewtonManager._adapter.step = spy("actuators", step)
    try:
        NewtonManager.set_decimation(GRAPH_DECIMATION)
        assert NewtonManager._graph is None, "the eager fixture must not capture a graph"
        events.clear()
        _release(robot)
        native_sim_eager.step()
    finally:
        MjWarpActuatorBridge.gather_external_torque = gather
        MjWarpActuatorBridge.publish_dof_friction = publish
        NewtonManager._adapter.step = step

    assert events == ["gather", "actuators", "publish"] * GRAPH_DECIMATION


def _build_two_native_pendulums(sim, pendulum_usd: str, second_cfg: BamActuatorCfg) -> tuple:
    """Spawn two BAM articulations per environment and initialize the simulation.

    Both robots are the same asset, so Newton merges their BAM joints into *one* actuator
    whenever the configurations agree on every grouping-key field -- which is exactly the
    multi-robot case the per-articulation binding has to get right.
    """
    for index in range(NUM_ENVS):
        sim_utils.create_prim(f"/World/Env_{index}", "Xform", translation=(index * 1.0, 0.0, 1.0))
    robots = []
    for name, cfg in (
        ("RobotA", BamActuatorCfg(joint_names_expr=[".*"], vin=VIN, kp_fw=KP_FW)),
        ("RobotB", second_cfg),
    ):
        robots.append(
            Articulation(
                ArticulationCfg(
                    prim_path=f"/World/Env_[^/]*/{name}",
                    spawn=sim_utils.UsdFileCfg(usd_path=pendulum_usd),
                    init_state=ArticulationCfg.InitialStateCfg(joint_pos={"joint": INITIAL_ANGLE}),
                    actuators={"servo": cfg},
                )
            )
        )
    sim.reset()
    return tuple(robots)


@pytest.mark.parametrize("device", test_devices())
def test_two_articulations_sharing_one_actuator_must_agree(native_sim, device, pendulum_usd):
    """Two robots merged into one Newton actuator cannot carry different BAM settings.

    ``vin_range``, ``vin_drop_gain_range``, ``friction_scale_range`` and ``stiff_frictionloss``
    are not part of Newton's actuator-grouping key, so structurally identical robots share one
    actuator and one set of parameter arrays. Applying the second articulation's ranges would
    silently discard the first's randomization, and skipping them would silently ignore the
    second's configuration, so the conflict has to be refused.
    """
    conflicting = BamActuatorCfg(joint_names_expr=[".*"], vin=VIN, kp_fw=KP_FW, friction_scale_range=(0.5, 2.0))
    with pytest.raises(ValueError, match="share one Newton actuator"):
        _build_two_native_pendulums(native_sim, pendulum_usd, conflicting)


@pytest.mark.parametrize("device", test_devices())
def test_two_articulations_with_matching_settings_bind_once(native_sim, device, pendulum_usd):
    """Agreeing robots share the actuator, and neither is left unbound or bound twice."""
    matching = BamActuatorCfg(joint_names_expr=[".*"], vin=VIN, kp_fw=KP_FW)
    robot_a, robot_b = _build_two_native_pendulums(native_sim, pendulum_usd, matching)

    bam_actuators = [
        actuator for actuator in NewtonManager._adapter.actuators if isinstance(actuator.controller, ControllerBam)
    ]
    assert len(bam_actuators) == 1, "the identical robots must merge into one Newton actuator"
    controller = bam_actuators[0].controller
    assert controller.solver_applies_friction
    assert controller.external_torque is not None

    # One binding, not one per articulation: a second registration would write the same budget
    # twice per step and, worse, hide a scoping mistake. The gather hook is the countable half
    # of the pair -- the publish hook is registered as a lambda, so it cannot be told apart from
    # any other post-actuator callback -- and both are registered together or not at all.
    assert len(NewtonManager._pre_actuator_callbacks) == 1

    for robot in (robot_a, robot_b):
        assert "servo" in robot.actuators._native_group_names


@pytest.mark.parametrize("device", test_devices())
def test_startup_ranges_are_sampled_per_environment(native_sim, device, pendulum_usd):
    """The config's start-up ranges must be drawn even though no solver exists yet.

    Sampling happens while the model is being built, before
    :meth:`~isaaclab_newton.physics.NewtonManager.initialize_solver` runs, precisely so that it
    does not depend on which solver the scene uses -- the values feed the controller's kernels,
    not the solver. Implementation A samples them on every backend and so must this one.
    """
    for index in range(NUM_ENVS):
        sim_utils.create_prim(f"/World/Env_{index}", "Xform", translation=(index * 1.0, 0.0, 1.0))
    robot = Articulation(
        ArticulationCfg(
            prim_path="/World/Env_[^/]*/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path=pendulum_usd),
            init_state=ArticulationCfg.InitialStateCfg(joint_pos={"joint": INITIAL_ANGLE}),
            actuators={
                "servo": BamActuatorCfg(
                    joint_names_expr=[".*"],
                    kp_fw=KP_FW,
                    vin_range=(6.0, 8.0),
                    friction_scale_range=(0.5, 1.5),
                )
            },
        )
    )
    native_sim.reset()
    assert robot.is_initialized

    for attr, (low, high) in (("vin", (6.0, 8.0)), ("friction_scale", (0.5, 1.5))):
        values = read_group_parameter(robot.actuators, "servo", "controller", attr)
        assert bool(((values >= low) & (values <= high)).all()), f"{attr} outside its configured range"
        assert len(torch.unique(values)) > 1, f"{attr} drew the same value for every environment"
    # An unset range keeps the authored nominal.
    torch.testing.assert_close(
        read_group_parameter(robot.actuators, "servo", "controller", "sag_gain"),
        torch.zeros(NUM_ENVS, robot.num_joints, device=robot.device),
    )


@pytest.mark.parametrize("device", test_devices())
def test_graph_capture_is_refused_at_a_decimation_of_one(native_sim, device, pendulum_usd):
    """Capturing a decimation of one with stateful actuators must fail loudly.

    The actuator state buffers are swapped host-side while the graph is recorded, so every
    replay restarts from the same buffer and the state never advances: BAM would report the
    friction budget of a freshly reset joint on every step. Silently wrong physics is worse
    than a refusal, so the refusal is the contract.
    """
    if not device.startswith("cuda"):
        pytest.skip("CUDA graph capture only happens on a CUDA device")
    _build_native_pendulum(native_sim, pendulum_usd)
    assert NewtonManager._adapter.is_stateful

    with pytest.raises(RuntimeError, match="decimation of one"):
        NewtonManager.set_decimation(1)

    # An even decimation is the documented way out, and it captures.
    NewtonManager.set_decimation(2)
    assert NewtonManager._graph is not None


@pytest.mark.parametrize("device", test_devices())
def test_each_articulation_configures_only_its_own_actuator(native_sim, device, pendulum_usd):
    """Two robots that do *not* merge must each get their own configuration.

    The actuator adapter is simulation-global, so an articulation that walked the whole
    adapter would apply its own start-up ranges to another robot's actuator -- silently, and
    with whichever articulation initialized first winning. Differing ``max_delay`` puts the two
    robots in different Newton actuators; only the scoping decides which one each configures.
    """
    delayed = BamActuatorCfg(
        joint_names_expr=[".*"], vin=VIN, kp_fw=KP_FW, max_delay=2, friction_scale_range=(3.0, 3.0)
    )
    robot_a, robot_b = _build_two_native_pendulums(native_sim, pendulum_usd, delayed)
    assert len({id(actuator) for actuator in NewtonManager._adapter.actuators}) == 2, (
        "the two robots must not merge, or the test cannot tell the configurations apart"
    )

    # robot_a keeps ``_build_two_native_pendulums``'s default cfg (no start-up range, so 1.0).
    for robot, expected in ((robot_a, 1.0), (robot_b, 3.0)):
        friction_scale = read_group_parameter(robot.actuators, "servo", "controller", "friction_scale")
        torch.testing.assert_close(friction_scale, torch.full_like(friction_scale, expected), atol=1e-6, rtol=0.0)


@pytest.mark.parametrize("device", test_devices())
def test_startup_ranges_are_sampled_without_a_mujoco_solver(device, pendulum_usd):
    """Start-up randomization must not depend on which solver the scene runs.

    The ranges feed the controller's kernels, not the solver, and implementation A samples them
    on every backend. On a solver that cannot apply joint dry friction the BAM controller falls
    back to its own stiction clip -- but the randomization is unaffected, which is only true
    because the sampling happens while the model is built, before any solver exists.
    """
    sim_cfg = SimulationCfg(
        dt=DT,
        device=device,
        use_newton_actuators=True,
        physics=NewtonCfg(solver_cfg=FeatherstoneSolverCfg(), num_substeps=2, debug_mode=False),
    )
    with build_simulation_context(
        device=device, gravity_enabled=True, add_ground_plane=False, sim_cfg=sim_cfg
    ) as sim_ctx:
        sim_ctx._app_control_on_stop_handle = None  # noqa: SLF001
        for index in range(NUM_ENVS):
            sim_utils.create_prim(f"/World/Env_{index}", "Xform", translation=(index * 1.0, 0.0, 1.0))
        robot = Articulation(
            ArticulationCfg(
                prim_path="/World/Env_[^/]*/Robot",
                spawn=sim_utils.UsdFileCfg(usd_path=pendulum_usd),
                init_state=ArticulationCfg.InitialStateCfg(joint_pos={"joint": INITIAL_ANGLE}),
                actuators={"servo": BamActuatorCfg(joint_names_expr=[".*"], kp_fw=KP_FW, vin_range=(6.0, 8.0))},
            )
        )
        sim_ctx.reset()
        assert robot.is_initialized

        controller = _native_controller(robot)
        # No MuJoCo model, so the controller keeps the torque-level clip ...
        assert not controller.solver_applies_friction
        assert controller.external_torque is None
        # ... and the start-up randomization happened anyway.
        vin = read_group_parameter(robot.actuators, "servo", "controller", "vin")
        assert bool(((vin >= 6.0) & (vin <= 8.0)).all())
        assert len(torch.unique(vin)) > 1
