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
import warp as wp
from isaaclab_newton.physics import (
    FeatherstoneSolverCfg,
    MjWarpActuatorBridge,
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonManager,
)

import isaaclab.sim as sim_utils
from isaaclab.actuators import BamActuator, BamActuatorCfg, BamBacklashActuatorCfg
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

BACKLASH_PENDULUM_USDA = """\
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

    def Xform "PlayDummy" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {
        float physics:mass = 1e-6
        float3 physics:diagonalInertia = (1e-9, 1e-9, 1e-9)
    }

    def Xform "PlayedArm" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {
        float physics:mass = 0.02
        point3f physics:centerOfMass = (0.05, 0, 0)
        float3 physics:diagonalInertia = (0.002, 0.002, 0.002)
    }

    def Xform "RigidArm" (
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

    def PhysicsRevoluteJoint "servo_played"
    {
        uniform token physics:axis = "Y"
        rel physics:body0 = </Robot/Pivot>
        rel physics:body1 = </Robot/PlayDummy>
        float physxJoint:armature = 0.0018
    }

    def PhysicsRevoluteJoint "passive_servo_played_backlash" (
        prepend apiSchemas = ["MjcJointAPI"]
    )
    {
        uniform token physics:axis = "Y"
        float physics:lowerLimit = -1
        float physics:upperLimit = 1
        rel physics:body0 = </Robot/PlayDummy>
        rel physics:body1 = </Robot/PlayedArm>
        float physxJoint:armature = 0.001
        float mjc:damping = 0.01
        double[] mjc:solimplimit = [0.95, 0.999, 0.0001, 0.5, 2]
        double[] mjc:solreflimit = [0.01, 1]
    }

    def PhysicsRevoluteJoint "servo_rigid"
    {
        uniform token physics:axis = "Y"
        rel physics:body0 = </Robot/Pivot>
        rel physics:body1 = </Robot/RigidArm>
        float physxJoint:armature = 0.0018
    }
}
"""
"""Two BAM servos on the same welded pivot, only one of which has modelled gear play.

``servo_played`` drives a massless dummy, and ``passive_servo_played_backlash`` -- the play hinge,
named by the convention :class:`~isaaclab.actuators.BamBacklashActuatorCfg` contracts on -- carries
the plus/minus one degree of gear play between that dummy and ``PlayedArm``. ``servo_rigid`` drives
its arm directly and has no such hinge, so one fixture covers both halves of the name lookup: a
servo that finds its twin and a servo that does not.

The dummy's mass and inertia are the converter's (1e-6 kg, 1e-9 kg m^2): the dead zone has to be a
free axis, not a second link. The inertia is deliberately *not* smaller. Newton's
``ModelBuilder.finalize`` validates inertia against an absolute eigenvalue floor of 1e-10 and adds
1e-6 when it corrects, so a 1e-12 dummy is silently inflated to a value three orders of magnitude
above the play hinge's own armature -- and warns once per body per environment while doing it.
1e-9 clears the floor untouched. Both servos carry the Dynamixel XL330's reflected rotor inertia as
joint armature, which is what makes the interval where the dummy hangs off nothing integrable at
all -- inside the dead zone the servo's only inertia is that armature.

The play hinge authors the MuJoCo limit-constraint parameters the converted asset does, because the
gear teeth *are* that constraint: at MuJoCo's default limit reference a range this small overshoots
roughly twofold, which would read as twice the play the model declares.
"""

FLOATING_BACKLASH_PENDULUM_USDA = BACKLASH_PENDULUM_USDA.replace(
    """    def PhysicsFixedJoint "anchor"
    {
        rel physics:body1 = </Robot/Pivot>
    }

""",
    "",
)
"""The play fixture with its anchor removed, i.e. on a floating base.

The robot every deployed backlash plant lives on is floating-base, and a free joint spends seven
positional coordinates against six degrees of freedom. The coordinate array the encoder binding is
read against and the velocity array its degrees of freedom are numbered in therefore drift apart by
one slot per environment, which an anchored fixture cannot show because there the two coincide.
Nothing is simulated on this variant -- it falls -- and nothing needs to be: what it exists to
measure is which slot the binding names.
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


def _build_native_pendulum(sim, pendulum_usd: str, actuator_cfg: BamActuatorCfg | None = None) -> Articulation:
    """Spawn :data:`NUM_ENVS` BAM-driven pendulums on the Newton-native actuator path.

    Args:
        sim: Simulation context to spawn into.
        pendulum_usd: Path of the pendulum asset.
        actuator_cfg: Servo group to drive them with. Defaults to the plain BAM configuration.
    """
    for index in range(NUM_ENVS):
        sim_utils.create_prim(f"/World/Env_{index}", "Xform", translation=(index * 1.0, 0.0, 1.0))
    robot = Articulation(
        ArticulationCfg(
            prim_path="/World/Env_[^/]*/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path=pendulum_usd),
            init_state=ArticulationCfg.InitialStateCfg(joint_pos={"joint": INITIAL_ANGLE}),
            actuators={"servo": actuator_cfg or BamActuatorCfg(joint_names_expr=[".*"], vin=VIN, kp_fw=KP_FW)},
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


"""
Encoder-through-backlash wiring (:class:`~isaaclab.actuators.BamBacklashActuatorCfg`).

The controller can already close its firmware loop on a second joint; what these tests cover is
everything between a task configuration and that binding. The Newton backend resolves each driven
joint's play hinge by name at articulation initialization, builds the per-DOF index and mask
arrays, and hands them to the controller before the first step -- and therefore before any CUDA
graph capture. The plant is the fixture above: one servo with a plus/minus one degree play hinge
and one without.
"""

SERVO_ONLY_EXPR = ["^(?!passive_).*"]
"""Group selection of the backlash assets: the servos, never the play hinges nothing drives."""

PLAYED_SERVO = "servo_played"
"""Fixture joint whose gearbox has modelled play."""

PLAY_HINGE = "passive_servo_played_backlash"
"""Fixture joint that carries :data:`PLAYED_SERVO`'s gear play."""

RIGID_SERVO = "servo_rigid"
"""Fixture joint whose gearbox has none, so the name lookup finds it no twin."""

PLAY_LIMIT = math.radians(1.0)
"""Half the gear play of the fixture's played servo [rad]."""

PLAY_WIDTH = 2.0 * PLAY_LIMIT
"""Peak-to-peak gear play of the fixture's played servo [rad], i.e. the dead zone's total width."""

PLATEAU_TOLERANCE = 0.2
"""Fraction of :data:`PLAY_WIDTH` the measured dead zone may differ from it by [-].

The gear teeth are a limit constraint, not a rigid stop, so the play reads slightly wide under
load. Twenty percent is the accuracy gate's own band for this signature.
"""

REVERSAL_TARGETS = (math.radians(70.0), math.radians(110.0))
"""Commanded angles either side of the arm's vertical [rad].

The joint spins about ``+Y`` and the arm's centre of mass sits at ``(L, 0, 0)``, so the gravity
torque is ``m*g*L*cos(theta)`` and reverses sign as the arm swings through ``pi/2``. Holding the
arm on one side and then the other is therefore a genuine reversal of the load the gearbox carries,
which is what makes the play hinge change limits -- the reversal the dead zone is measured across.
Twenty degrees off vertical keeps the load well clear of zero without leaving the servo's reach.
"""

REVERSAL_STEPS = 300
"""Steps each side of the reversal settles for [-]. The arm is at rest well before this."""

DEAD_ZONE_FRICTION_SCALE = (0.02, 0.02)
"""Friction-budget scale the dead-zone measurement runs the gearbox at [-].

At the identified budget this servo's stiction band is about a degree wide in measured position --
the same order as the play itself -- so the joint stops as soon as friction can hold it and where
it stopped says nothing about which quantity the firmware was regulating. A fiftieth of the budget
narrows the band to well under a tenth of the play. It is a measurement condition and not a plant
change: the play hinge, its limits and its limit constraint are untouched, and the dead zone this
measures is the same one a fully frictional gearbox has.
"""


@pytest.fixture(scope="module")
def backlash_pendulum_usd(tmp_path_factory) -> str:
    """Write :data:`BACKLASH_PENDULUM_USDA` to a temporary file and return its path."""
    path = tmp_path_factory.mktemp("bam_backlash") / "backlash_pendulum.usda"
    path.write_text(BACKLASH_PENDULUM_USDA)
    return str(path)


@pytest.fixture(scope="module")
def floating_backlash_pendulum_usd(tmp_path_factory) -> str:
    """Write :data:`FLOATING_BACKLASH_PENDULUM_USDA` to a temporary file and return its path."""
    path = tmp_path_factory.mktemp("bam_backlash_floating") / "floating_backlash_pendulum.usda"
    path.write_text(FLOATING_BACKLASH_PENDULUM_USDA)
    return str(path)


def _backlash_robot_cfg(actuator_cfg) -> ArticulationCfg:
    """Return the fixture's articulation configuration, driven by *actuator_cfg*."""
    return ArticulationCfg(
        prim_path="/World/Env_[^/]*/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=None),
        init_state=ArticulationCfg.InitialStateCfg(),
        actuators={"servo": actuator_cfg},
    )


def _build_backlash_pendulum(sim, usd_path: str, actuator_cfg) -> Articulation:
    """Spawn :data:`NUM_ENVS` copies of the play fixture and initialize the simulation."""
    for index in range(NUM_ENVS):
        sim_utils.create_prim(f"/World/Env_{index}", "Xform", translation=(index * 1.0, 0.0, 1.0))
    cfg = _backlash_robot_cfg(actuator_cfg)
    cfg.spawn.usd_path = usd_path
    robot = Articulation(cfg)
    sim.reset()
    assert robot.is_initialized
    return robot


def _backlash_actuator_cfg(**overrides) -> BamBacklashActuatorCfg:
    """Return the fixture's encoder-through-play servo group."""
    kwargs = {"joint_names_expr": SERVO_ONLY_EXPR, "vin": VIN, "kp_fw": KP_FW}
    kwargs.update(overrides)
    return BamBacklashActuatorCfg(**kwargs)


def _settle_at(robot: Articulation, sim, angle: float, steps: int = REVERSAL_STEPS) -> torch.Tensor:
    """Release both servos at *angle* with the play closed, hold it as the target, and settle.

    Returns:
        The recorded joint positions [rad], shape ``(steps, NUM_ENVS, num_joints)``.
    """
    released = torch.zeros_like(robot.data.joint_pos.torch)
    for name in _servo_names(robot):
        released[:, robot.joint_names.index(name)] = angle
    robot.write_joint_position_to_sim_index(position=released)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(robot.data.joint_vel.torch))
    robot.actuators.reset()
    robot.actuators.target_command.set_position_index(value=released.clone())

    positions = []
    for _ in range(steps):
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())
        positions.append(robot.data.joint_pos.torch.clone())
    trace = torch.stack(positions)
    assert torch.isfinite(trace).all(), "non-finite joint position in the rollout"
    return trace


def _servo_names(robot: Articulation) -> list[str]:
    """Return the fixture's driven joints, i.e. everything the play hinges are not."""
    return [name for name in robot.joint_names if not name.startswith("passive_")]


def _joint_angle(trace: torch.Tensor, robot: Articulation, name: str) -> torch.Tensor:
    """Return one joint's final angle in every environment [rad], shape ``(NUM_ENVS,)``."""
    return trace[-1, :, robot.joint_names.index(name)]


def _unbind_backlash(controller: ControllerBam, device: str) -> None:
    """Rebind *controller* with an all-zero mask, i.e. back to the plain servo."""
    controller.bind_backlash_indices(
        wp.array(controller.backlash_pos_indices.numpy(), dtype=wp.uint32, device=device),
        wp.zeros(len(controller.backlash_mask), dtype=wp.float32, device=device),
    )


def _scene_bam_actuator():
    """Return the scene's single BAM actuator."""
    actuators = [
        actuator for actuator in NewtonManager._adapter.actuators if isinstance(actuator.controller, ControllerBam)
    ]
    assert len(actuators) == 1, "the fixture has exactly one BAM actuator"
    return actuators[0]


def _actuator_dof_indices() -> list[int]:
    """Return the flat model DOF indices the scene's single BAM actuator drives."""
    return [int(index) for index in _scene_bam_actuator().indices.numpy()]


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("base", ("anchored", "floating"))
def test_each_servo_is_bound_to_the_play_hinge_in_series_with_it(
    native_sim, device, base, backlash_pendulum_usd, floating_backlash_pendulum_usd
):
    """The name lookup must produce the controller's per-DOF binding, DOF by DOF.

    The binding is a pair of flat indices into the *whole* model's position array, in the
    actuator's own DOF order (environment major). Everything about it is easy to get subtly
    wrong -- the stride, the environment offset, the articulation's own offset -- and every
    mistake reads a real angle from some other joint, so the expectation here is built from the
    articulation's joint names and the model's own coordinate count rather than from the resolver.

    The array is indexed into *positions*, so the layout it must be numbered in is the coordinate
    one, and that is not the layout the actuator's degrees of freedom are numbered in: a free joint
    spends seven coordinates and six degrees of freedom, so the two drift by one slot per
    environment. The parametrization runs the same expectation on both bases, because on an
    anchored articulation the two layouts coincide and any confusion between them is invisible.

    A servo whose plant models no play takes a zero mask, and its index is then never
    dereferenced. It is still filled with the DOF's *own* position slot: the array is
    self-documenting that way, and neither slot zero nor a plain ``arange`` would be the self
    index once an actuator covers a subset of the articulation's joints, which is exactly what a
    servo group on a plant with play hinges does.
    """
    usd_path = backlash_pendulum_usd if base == "anchored" else floating_backlash_pendulum_usd
    robot = _build_backlash_pendulum(native_sim, usd_path, _backlash_actuator_cfg())
    controller = _native_controller(robot)
    joint_names = robot.backend_joint_names
    assert set(joint_names) == {PLAYED_SERVO, PLAY_HINGE, RIGID_SERVO}

    model = NewtonManager._model
    dofs_per_env = model.joint_dof_count // NUM_ENVS
    coords_per_env = model.joint_coord_count // NUM_ENVS
    # Whatever the root joint spends ahead of the articulation's own hinges, in each layout.
    root_dofs = dofs_per_env - len(joint_names)
    root_coords = coords_per_env - len(joint_names)
    assert (root_dofs, root_coords) == ((0, 0) if base == "anchored" else (6, 7)), (
        "the fixture is the only articulation in the model, on the base under test"
    )
    slots = {name: joint_names.index(name) for name in joint_names}

    # The actuator drives the two servos and nothing else, in environment-major order.
    driven = [(index // dofs_per_env, index % dofs_per_env - root_dofs) for index in _actuator_dof_indices()]
    assert driven == [(env, slots[name]) for env in range(NUM_ENVS) for name in (PLAYED_SERVO, RIGID_SERVO)] or (
        driven == [(env, slots[name]) for env in range(NUM_ENVS) for name in (RIGID_SERVO, PLAYED_SERVO)]
    ), "the actuator must cover exactly the two servos, environment major"

    def coordinate_of(env: int, slot: int) -> int:
        return env * coords_per_env + root_coords + slot

    # Newton's own numbering of the servos' positions, which is what says the formula above is the
    # coordinate layout rather than a restatement of the resolver's arithmetic.
    np.testing.assert_array_equal(
        _scene_bam_actuator().pos_indices.numpy(),
        np.array([coordinate_of(env, slot) for env, slot in driven], dtype=np.uint32),
    )

    expected_mask = []
    expected_indices = []
    for env, slot in driven:
        played = slot == slots[PLAYED_SERVO]
        expected_mask.append(1.0 if played else 0.0)
        expected_indices.append(coordinate_of(env, slots[PLAY_HINGE] if played else slot))

    np.testing.assert_array_equal(controller.backlash_mask.numpy(), np.array(expected_mask, dtype=np.float32))
    np.testing.assert_array_equal(controller.backlash_pos_indices.numpy(), np.array(expected_indices, dtype=np.uint32))
    # The unplayed servo's entry is its own slot, which on this fixture is neither 0 nor its
    # position in the actuator's DOF order -- the two fillings the contract rules out.
    assert expected_indices != list(range(len(expected_indices)))


@pytest.mark.parametrize("device", test_devices())
def test_the_play_opens_a_two_degree_dead_zone_across_a_load_reversal(native_sim, device, backlash_pendulum_usd):
    """Hold the arm either side of vertical and measure the dead zone the gear play opens.

    Rotating this arm through ``pi/2`` reverses the sign of the gravity torque it hangs from, so
    holding it at 70 and then at 110 degrees is a genuine reversal of the load the gearbox
    carries -- and a gearbox with play answers a load reversal by crossing its dead zone. The
    signature is a plateau: the motor travels the full peak-to-peak play *more* than the encoder
    reads, because that much of its travel goes into taking up the teeth rather than into moving
    the link. Both settled states are quasi-static, so the plateau is read off two resting states
    rather than chased through a transient.

    What is under test here is the wiring, not the hinge: the play swing is a property of the
    plant and would be there whatever the servo believes. The binding is what decides *which*
    quantity the firmware regulates, and the last assertion is the one that reads it -- with the
    encoder bound the link angle lands on the commanded target and the motor sits a play-width
    away from it, and without the binding the two swap round.
    """
    robot = _build_backlash_pendulum(
        native_sim, backlash_pendulum_usd, _backlash_actuator_cfg(friction_scale_range=DEAD_ZONE_FRICTION_SCALE)
    )
    settled = {}
    for target in REVERSAL_TARGETS:
        trace = _settle_at(robot, native_sim, target)
        servo = _joint_angle(trace, robot, PLAYED_SERVO)
        play = _joint_angle(trace, robot, PLAY_HINGE)
        settled[target] = (servo, play, servo + play)
    (servo_low, play_low, measured_low), (servo_high, play_high, measured_high) = settled.values()

    # The precondition the whole measurement rests on: the load really did reverse. The arm's
    # gravity torque is ``m*g*L*cos(theta)``, so the two states have to straddle ``pi/2``.
    assert bool((torch.cos(measured_low) > 0).all()), "the first state is not on the near side of vertical"
    assert bool((torch.cos(measured_high) < 0).all()), "the second state is not on the far side of vertical"

    # The play crossed its dead zone with the load, ending on the opposite tooth each time.
    play_tolerance = PLATEAU_TOLERANCE * PLAY_LIMIT
    torch.testing.assert_close(play_low, torch.full_like(play_low, PLAY_LIMIT), atol=play_tolerance, rtol=0.0)
    torch.testing.assert_close(play_high, torch.full_like(play_high, -PLAY_LIMIT), atol=play_tolerance, rtol=0.0)

    # The plateau: motor travel the encoder never saw, over the reversal.
    plateau = (servo_high - servo_low).abs() - (measured_high - measured_low).abs()
    torch.testing.assert_close(
        plateau, torch.full_like(plateau, PLAY_WIDTH), atol=PLATEAU_TOLERANCE * PLAY_WIDTH, rtol=0.0
    )

    # ... and the firmware closed its loop on the encoder, not on its own shaft: the link lands on
    # the target and the motor is the one sitting a play-width off. An unbound controller settles
    # the other way round, which is what makes this the assertion that reads the binding.
    for target, (servo, _, measured) in settled.items():
        assert float((measured - target).abs().max()) < float((servo - target).abs().min()), (
            f"the motor, not the encoder, tracked the {math.degrees(target):.0f} degree target"
        )


@pytest.mark.parametrize("device", test_devices())
def test_a_bound_backlash_articulation_is_captured_in_the_cuda_graph(native_sim, device, backlash_pendulum_usd):
    """The encoder view must survive the decimation loop's CUDA graph capture, read live.

    A controller that forced eager stepping would cost the whole decimation loop its capture, so
    the property is asserted rather than assumed on a real articulation with real play hinges --
    the unit suite proves the kernel captures, this proves the wired-up plant does. Capture
    happens on :meth:`~isaaclab_newton.physics.NewtonManager.set_decimation`, which an
    environment calls for its policy decimation and a bare simulation context never does.

    Replaying under a *changed* binding is the real evidence, exactly as the friction-budget
    capture test replays under a changed randomization: the arrays the controller was finalized
    with are the ones the recorded launch reads, so dropping the mask has to move the settled
    motor angle by the play. A binding baked into the graph, or one rebound by replacing the
    arrays instead of copying into them, leaves it where it was.
    """
    robot = _build_backlash_pendulum(
        native_sim, backlash_pendulum_usd, _backlash_actuator_cfg(friction_scale_range=DEAD_ZONE_FRICTION_SCALE)
    )
    controller = _native_controller(robot)
    assert NewtonManager._adapter.is_all_graphable
    assert bool(controller.backlash_mask.numpy().any()), "the fixture's played servo must be bound"

    NewtonManager.set_decimation(GRAPH_DECIMATION)
    if device.startswith("cuda"):
        assert NewtonManager._graph is not None, "the decimation loop was not captured"

    bound = _joint_angle(_settle_at(robot, native_sim, REVERSAL_TARGETS[0]), robot, PLAYED_SERVO)
    _unbind_backlash(controller, robot.device)
    unbound = _joint_angle(_settle_at(robot, native_sim, REVERSAL_TARGETS[0]), robot, PLAYED_SERVO)

    # Dropping the mask hands the firmware its own shaft angle again, so the motor settles a
    # play-width away from where the encoder view put it.
    separation = float((bound - unbound).abs().min())
    assert separation > 0.5 * PLAY_LIMIT, f"the replayed graph kept the old binding ({separation:.2e} rad)"


@pytest.mark.parametrize("device", test_devices())
def test_a_plant_without_play_hinges_degrades_to_the_plain_servo(device, pendulum_usd):
    """A backlash configuration on a model with no play hinges must cost exactly nothing.

    The name lookup finds no twin for the pendulum's one joint, which is the documented
    degrade-to-plain case: mask zero, and the index left at the DOF's own position slot because
    the kernel then never dereferences it. What that has to buy is not "almost the plain servo"
    but the plain servo, so the two rollouts are compared with no tolerance at all -- a plant a
    policy was trained against may not shift under a configuration change that models nothing.
    """
    rollouts: dict[str, torch.Tensor] = {}
    bindings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, actuator_cfg in (
        ("plain", BamActuatorCfg(joint_names_expr=[".*"], vin=VIN, kp_fw=KP_FW)),
        ("backlash", BamBacklashActuatorCfg(joint_names_expr=[".*"], vin=VIN, kp_fw=KP_FW)),
    ):
        with build_simulation_context(
            device=device,
            gravity_enabled=True,
            add_ground_plane=False,
            sim_cfg=_make_sim_cfg(device, use_newton_actuators=True),
        ) as sim_ctx:
            sim_ctx._app_control_on_stop_handle = None  # noqa: SLF001
            robot = _build_native_pendulum(sim_ctx, pendulum_usd, actuator_cfg)
            controller = _native_controller(robot)
            bindings[name] = (
                controller.backlash_mask.numpy().copy(),
                controller.backlash_pos_indices.numpy().copy(),
            )
            dof_indices = np.array(_actuator_dof_indices(), dtype=np.uint32)
            _release(robot)
            rollouts[name] = _settle(robot, sim_ctx)[0].cpu()

    mask, indices = bindings["backlash"]
    assert not mask.any(), "a joint with no play hinge must be masked off"
    np.testing.assert_array_equal(indices, dof_indices, err_msg="the masked-off index is not the DOF's own slot")
    np.testing.assert_array_equal(
        bindings["plain"][0], mask, err_msg="the plain configuration must leave the same all-zero mask"
    )
    torch.testing.assert_close(rollouts["backlash"], rollouts["plain"], atol=0.0, rtol=0.0)


@pytest.mark.parametrize("device", test_devices())
def test_the_backlash_cfg_is_refused_on_the_isaac_lab_executed_path(sim, device, pendulum_usd):
    """``use_newton_actuators=False`` must refuse the configuration rather than drop the play.

    This fixture's simulation runs the Isaac Lab actuator loop, which is handed one group's
    joints and cannot read the play hinge beside them. There is no degraded mode to fall back
    to, so the refusal names the one-line fix instead of quietly training a policy against a
    plant without the play its configuration asked for.
    """
    with pytest.raises(ValueError, match="use_newton_actuators"):
        _build_native_pendulum(sim, pendulum_usd, BamBacklashActuatorCfg(joint_names_expr=[".*"], kp_fw=KP_FW))
