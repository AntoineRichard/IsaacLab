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

import pytest
import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import BamActuator, BamActuatorCfg
from isaaclab.actuators.bam_model import (
    BamMotorParams,
    compute_duty,
    compute_friction_budget,
    compute_motor_torque,
)
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


def _make_sim_cfg(device: str) -> SimulationCfg:
    """Build the MJWarp configuration used by every test in this module."""
    return SimulationCfg(
        dt=DT,
        device=device,
        # The explicit Python actuator path is the one under test.
        use_newton_actuators=False,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(njmax=20, nconmax=20, ls_iterations=20, integrator="implicitfast", impratio=1),
            num_substeps=2,
            debug_mode=False,
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
    """Newton simulation context for the requested device."""
    with build_simulation_context(
        device=device,
        gravity_enabled=True,
        add_ground_plane=False,
        sim_cfg=_make_sim_cfg(device),
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


def _assert_rest(positions: torch.Tensor, velocities: torch.Tensor, efforts: torch.Tensor) -> torch.Tensor:
    """Assert the rollout is finite and has come to rest, and return the final angles."""
    for name, trace in (("position", positions), ("velocity", velocities), ("effort", efforts)):
        assert torch.isfinite(trace).all(), f"non-finite joint {name} in the rollout"
    assert velocities[-1].abs().max() < 1e-3, "the pendulum has not come to rest"
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
