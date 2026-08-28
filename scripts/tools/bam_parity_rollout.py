# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Roll a pendulum through Isaac Lab's BAM actuator and the reference BAM simulator, and compare.

The reference is the CPU simulator the BAM paper validated against bench logs
(``bam/simulate.py``, :class:`bam.simulate.Simulator`): a single-axis pendulum whose firmware,
motor, friction and integration are all evaluated in numpy. Isaac Lab computes the same servo
model but hands its torque to a Newton/MJWarp articulation, which owns the rigid-body dynamics
instead. This harness drives both with the *same* position trajectory and reports how far apart
the two joint angles drift.

The two implementations under test
----------------------------------
``--impl`` selects which port is compared against the reference. Both are built from the same
:class:`~isaaclab.actuators.BamActuatorCfg` on the same fixture; the switch between them is
:attr:`~isaaclab.sim.SimulationCfg.use_newton_actuators`:

``lab``
    **Implementation A**, :class:`~isaaclab.actuators.BamActuator`: the model runs in Python
    between the physics steps and writes a joint-effort command. It applies BAM's static
    friction itself, by clipping the torque, and it *estimates* the external load from the
    rotor momentum balance. This is the structure the reference has, so the comparison is
    between two spellings of one algorithm.
``newton``
    **Implementation B**, :class:`~isaaclab.actuators.newton.ControllerBam`: the model runs as
    a Warp controller inside the Newton actuator path. It publishes the friction budget into
    MuJoCo's ``dof_frictionloss`` and lets the solver clip, and it *reads* the external load
    out of the solver's generalized forces. Both differences are real modelling changes
    against the reference and are what this run measures; neither is a bug to be tuned away.
    Consequently ``efforts`` means something different on this path -- see
    :class:`ActuatorRollout` -- and B's total error is **not** required to be below A's.

What is matched, and what is not
--------------------------------
Everything that can be matched is matched, so the residual is the modelling gap and not a
setup difference:

* **Plant.** The reference pendulum is a point mass ``m`` at radius ``L`` plus the reflected
  rotor inertia (``bam.testbench.Pendulum`` + ``Actuator.get_extra_inertia``), giving
  ``I = m L^2 + armature`` and a gravity torque ``m g L sin(q)`` with ``g = -9.80665``. The
  Isaac Lab side authors exactly that: an arm of mass ``m`` whose centre of mass sits at
  ``(0, 0, -L)`` -- so ``q = 0`` hangs straight down, as upstream defines it -- and whose
  inertia *about its centre of mass* is the armature, so the parallel-axis theorem puts
  ``m L^2 + armature`` about the joint axis. The simulation gravity is set to the same
  ``9.80665`` that ``bam.testbench`` hard-codes. The ``plant`` ablation measures both sides'
  effective inertia and refuses to report anything if they disagree.
* **Parameters.** Both sides are built from the *same* JSON: Isaac Lab's vendored
  ``bam_xl330_m6.json`` is fed to :class:`~isaaclab.actuators.BamActuatorCfg` and to upstream's
  own :func:`bam.model.load_model_from_dict`. A tripwire checks that the firmware constants the
  file carries (``error_gain``, ``max_pwm``, ``max_current``) are the ones upstream keeps in
  code, so the two cannot silently disagree.
* **Configuration.** ``kp_fw = 200``, a fixed ``vin = 7.4`` with no supply sag, no command
  delay and ``friction_scale = 1`` -- the domain-randomization machinery is switched off so the
  comparison isolates the plant and the actuator.
* **Deliberately unmatched.** ``q_offset`` (the identification testbench's mounting error) is
  zeroed on the reference: the Isaac Lab pendulum has no such offset, and the Isaac Lab port
  does not model one.
* **Unmatchable.** The integrators. Isaac Lab runs MJWarp's ``implicitfast`` with one substep,
  which advances the position by ``dq_next * dt``; ``Simulator.step`` adds a further
  ``0.5 * ddq * dt^2`` on top of that same update. The extra half-step is a systematic
  position bias of order ``dt`` per unit time, and it is the residual the ``plant`` ablation
  isolates. Sweeping ``--dt`` shows the rest of the comparison inheriting the same first-order
  scaling.

The deviations under measurement
--------------------------------
:class:`~isaaclab.actuators.BamActuator` runs *outside* the solver, which costs it two things
the reference gets for free:

1. **The external torque is estimated, not read.** Upstream passes the true bias torque into
   the friction budget and the stopping-torque test; Isaac Lab reconstructs it from the rotor
   momentum balance, ``armature * ddq - tau_applied_prev``, which is exact only while the link
   is not accelerating. ``ext_torque_*`` reports that residual against the load the pendulum
   really applies, and ``budget_error_*`` reports how much of it survives into the friction
   budget -- the only channel through which it can move the simulation.
2. **The stiction clip is sized with the rotor inertia alone.** Upstream's stopping torque uses
   the full ``m L^2 + armature``; the actuator only knows its own armature, so it
   underestimates the torque needed to arrest the joint by the link's share (reported as
   ``link_inertia_fraction``).

:class:`~isaaclab.actuators.newton.ControllerBam` closes both -- it reads the load, and MuJoCo's
friction-loss constraint arrests the joint with the true articulated inertia -- and pays for
them with a third deviation the reference does not have:

3. **The friction is a constraint, not a torque clip.** ``dof_frictionloss`` is enforced by the
   solver, jointly with everything else, and MuJoCo's friction-loss constraint is compliant: a
   "held" joint creeps instead of stopping dead. The same two keys stay meaningful --
   ``ext_torque_*`` is now the residual of the *read* load, which is exact but one physics step
   old, and ``budget_error_*`` what remains of it in the budget -- so the two implementations
   are decomposed onto one scale, and the constraint's compliance is what is left over.

Ablations
---------
``--ablation`` selects which terms are active, on **both** sides simultaneously. The BAM
``m1``--``m6`` model family is the switch, so no equation is edited on either side:

``full``
    ``m6`` -- everything on. The headline number.
``no-load-friction``
    ``m2`` -- Coulomb + Stribeck friction only. The load-dependent budget vanishes, so the
    estimated external torque no longer feeds the friction and only its (much weaker) path
    through the stopping-torque test survives.
``plant``
    ``m1`` with ``kt``, ``friction_base`` and ``friction_viscous`` zeroed, released from
    :data:`PLANT_INITIAL_ANGLE`. Both actuators produce exactly zero torque, so this compares
    the two *rigid-body* models alone: it validates the fixture through the measured effective
    inertia, and its position error is the pure integrator floor of the comparison.

Usage
-----
.. code-block:: bash

    uv run --with /path/to/checkouts/bam python scripts/tools/bam_parity_rollout.py \
        --impl lab --ablation full

The run aborts unless the installed ``bam`` really is the pinned reference revision; a local
checkout is accepted as long as its ``git`` HEAD matches and its working tree is clean, so the
printed report can never claim a revision it did not run. The pin and the check both live in
the sibling ``generate_bam_goldens.py``, so the goldens and this harness cannot drift apart.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from bam.actuator import VoltageControlledActuator
from bam.model import Model, load_model_from_dict
from bam.simulate import Simulator

# Running this file by path puts ``scripts/tools`` on ``sys.path``, so the sibling generator
# that owns the reference revision pin -- and the guard that enforces it -- is importable.
# Sharing both keeps the goldens and this harness from drifting onto different ``bam`` commits
# or onto two variants of one provenance policy.
from generate_bam_goldens import resolve_installed_bam_revision

from isaaclab.actuators import BAM_XL330_M6_PARAMS_FILE, BamActuatorCfg
from isaaclab.actuators.bam_model import (
    BamMotorParams,
    apply_stiction_clip,
    compute_duty,
    compute_friction_budget,
    compute_motor_torque,
    compute_stribeck_coeff,
)

# Rollout definition -- a linear chirp swept over the servo's usable band.
DT = 0.005
"""Control and physics timestep [s]."""

CHIRP_DURATION = 20.0
"""Time the chirp takes to sweep its whole frequency range [s]."""

CHIRP_START_FREQUENCY = 0.1
"""Instantaneous frequency of the goal trajectory at ``t = 0`` [Hz]."""

CHIRP_END_FREQUENCY = 2.0
"""Instantaneous frequency of the goal trajectory at ``t = CHIRP_DURATION`` [Hz]."""

CHIRP_AMPLITUDE = 0.6
"""Amplitude of the goal trajectory [rad]."""

NUM_PHASES = 4
"""Equal-length windows the rollout is broken into for the per-phase breakdown [-]."""

# Pendulum, matching ``bam.testbench.Pendulum`` with ``arm_mass = 0``.
PENDULUM_MASS = 0.1
"""Point mass at the tip of the arm [kg]."""

PENDULUM_LENGTH = 0.1
"""Distance from the joint axis to the mass [m]."""

GRAVITY = 9.80665
"""Gravitational acceleration [m/s^2]. ``bam.testbench.Pendulum`` hard-codes this value."""

PLANT_INITIAL_ANGLE = 0.6
"""Angle the arm is released from in the ``plant`` ablation [rad]."""

INERTIA_MISMATCH_TOLERANCE = 1e-5
"""Largest relative disagreement allowed between the two sides' effective inertia [-]."""

REPLAY_TOLERANCE = 1e-5
"""Largest deviation allowed when replaying the actuator pipeline over its own rollout [N.m].

The replay runs in double precision over a single-precision rollout, so it can never match the
recorded efforts exactly; the tolerance only has to be tight enough to catch a replay that
stopped describing the same model.
"""

# Actuator configuration -- fixed supply, nominal firmware gain, no randomization.
KP_FW = 200.0
"""Firmware proportional gain [-]."""

VIN = 7.4
"""Supply voltage [V]."""

# Divergence guards. The reference clips its own velocity at 100 rad/s, so a rollout that
# reaches either bound is no longer tracking anything.
MAX_PLAUSIBLE_ANGLE = 4.0 * CHIRP_AMPLITUDE
"""Largest joint angle a rollout may reach before it counts as diverged [rad]."""

MAX_PLAUSIBLE_VELOCITY = 100.0
"""Largest joint velocity a rollout may reach before it counts as diverged [rad/s]."""

PENDULUM_USDA_TEMPLATE = """\
#usda 1.0
(
    defaultPrim = "Robot"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Robot" (
    prepend apiSchemas = ["PhysicsArticulationRootAPI"]
)
{{
    def Xform "Pivot" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {{
        float physics:mass = 0.1
        float3 physics:diagonalInertia = (0.001, 0.001, 0.001)
    }}

    def Xform "Arm" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {{
        float physics:mass = {mass}
        point3f physics:centerOfMass = (0, 0, {com_z})
        float3 physics:diagonalInertia = ({inertia}, {inertia}, {inertia})
    }}

    def PhysicsFixedJoint "anchor"
    {{
        rel physics:body1 = </Robot/Pivot>
    }}

    def PhysicsRevoluteJoint "joint"
    {{
        uniform token physics:axis = "Y"
        rel physics:body0 = </Robot/Pivot>
        rel physics:body1 = </Robot/Arm>
    }}
}}
"""
"""Single-degree-of-freedom pendulum equivalent to ``bam.testbench.Pendulum``.

``Pivot`` is welded to the world by a fixed joint whose ``physics:body0`` is left unset; that
weld is what makes the USD physics parser emit an articulation at all (a lone body hanging off
a world-anchored revolute joint imports as an orphan joint).

``Arm`` carries the whole mass at ``(0, 0, -L)`` and the joint spins about ``+Y``, so ``q = 0``
points the arm straight down and gravity applies ``-m g L sin(q)`` about the axis -- the sign
and the zero of ``Pendulum.compute_bias``. Its inertia is authored *about the centre of mass*
and set to the reflected rotor inertia, so the parallel-axis theorem gives ``m L^2 + armature``
about the joint: exactly the inertia ``Simulator.step`` integrates with.
"""


@dataclass(frozen=True)
class Ablation:
    """One comparison configuration, applied identically to both simulators.

    Attributes:
        name: Command-line name of the ablation.
        description: One-line summary printed with the results.
        overrides: Keys written over the vendored BAM parameter file before either side is
            built. ``"model"`` selects a member of the BAM ``m1``--``m6`` family, which is how
            friction terms are switched off without editing an equation on either side.
        initial_angle: Angle both pendulums start from [rad].
        dead_motor: Whether the configuration produces exactly zero actuator torque. When set,
            the goal is held at the release angle and the harness additionally measures each
            side's effective joint inertia, which is only observable while nothing but gravity
            acts on the joint.
    """

    name: str
    description: str
    overrides: dict[str, Any] = field(default_factory=dict)
    initial_angle: float = 0.0
    dead_motor: bool = False


ABLATIONS: dict[str, Ablation] = {
    "full": Ablation(
        name="full",
        description="m6: every friction term active (the headline comparison)",
    ),
    "no-load-friction": Ablation(
        name="no-load-friction",
        description="m2: Coulomb + Stribeck only, no load-dependent friction on either side",
        overrides={"model": "m2"},
    ),
    "plant": Ablation(
        name="plant",
        description="m1 with a dead motor and no friction: rigid-body models only",
        overrides={"model": "m1", "kt": 0.0, "friction_base": 0.0, "friction_viscous": 0.0},
        initial_angle=PLANT_INITIAL_ANGLE,
        dead_motor=True,
    ),
}


@dataclass(frozen=True)
class Rollout:
    """The trajectory both simulators are driven with.

    Attributes:
        dt: Control and physics timestep [s].
        initial_angle: Angle both pendulums start from [rad].
        goal: Commanded joint position at every step [rad], shape ``(num_steps,)``.
        is_chirp: Whether ``goal`` is the frequency sweep. False for a held target, whose
            windows carry no frequency to report.
    """

    dt: float
    initial_angle: float
    goal: np.ndarray
    is_chirp: bool

    @property
    def num_steps(self) -> int:
        """Number of steps in the rollout [-]."""
        return len(self.goal)

    def instantaneous_frequency(self, step: int) -> float | None:
        """Return the chirp frequency at ``step`` [Hz], or None when the target is held."""
        if not self.is_chirp:
            return None
        sweep_rate = (CHIRP_END_FREQUENCY - CHIRP_START_FREQUENCY) / CHIRP_DURATION
        return CHIRP_START_FREQUENCY + sweep_rate * step * self.dt


def _require(condition: bool, message: str) -> None:
    """Raise :class:`RuntimeError` when ``condition`` is false.

    Used instead of ``assert`` so the checks still run under ``python -O``: a parity report that
    silently skipped its own validation would be worse than no report.
    """
    if not condition:
        raise RuntimeError(message)


def build_parameters(ablation: Ablation) -> dict[str, Any]:
    """Return the BAM parameter dictionary both simulators are built from.

    Starts from the parameter file Isaac Lab vendors -- the same one the actuator uses in
    production -- and applies the ablation's overrides. ``q_offset``, the identification
    testbench's mounting error, is always zeroed: the Isaac Lab pendulum has no such offset, and
    the Isaac Lab port does not model one.
    """
    parameters = json.loads(Path(BAM_XL330_M6_PARAMS_FILE).read_text())
    parameters["q_offset"] = 0.0
    parameters["kp"] = KP_FW
    parameters["vin"] = VIN
    parameters.update(ablation.overrides)
    return parameters


def build_reference_model(parameters: dict[str, Any]) -> Model:
    """Build the upstream BAM model from ``parameters`` and check it against the file.

    :func:`bam.model.load_model_from_dict` takes the fitted friction and motor terms from the
    dictionary but the firmware constants from the actuator class. Isaac Lab's parameter file
    carries copies of those constants, so they are compared here: if upstream ever re-tunes
    ``error_gain``, ``max_pwm`` or ``max_current``, this fails instead of quietly comparing two
    differently-configured servos.
    """
    model = load_model_from_dict(parameters)
    actuator = model.actuator
    _require(
        isinstance(actuator, VoltageControlledActuator),
        f"Expected a VoltageControlledActuator, got {type(actuator).__name__}.",
    )
    firmware = {"error_gain": actuator.error_gain, "max_pwm": actuator.max_pwm, "max_current": actuator.max_current}
    for name, reference_value in firmware.items():
        _require(
            reference_value == parameters[name],
            f"Firmware constant {name!r} differs between the reference actuator ({reference_value!r}) and the"
            f" parameter file Isaac Lab is configured from ({parameters[name]!r}).",
        )
    for name in ("kt", "R", "armature"):
        _require(
            getattr(model, name).value == parameters[name],
            f"Fitted parameter {name!r} was not loaded into the reference model.",
        )
    model.q_offset.value = parameters["q_offset"]
    return model


def build_rollout(ablation: Ablation, dt: float, duration: float) -> Rollout:
    """Return the commanded trajectory for one run.

    A linear chirp sweeps the instantaneous frequency from :data:`CHIRP_START_FREQUENCY` to
    :data:`CHIRP_END_FREQUENCY` over :data:`CHIRP_DURATION`, so one run covers the quasi-static
    regime (where the friction budget decides the answer) and the resonant one (where the link's
    inertia does). It starts at zero, which is where both pendulums hang, so nothing is excited
    by a step. A ``duration`` below :data:`CHIRP_DURATION` truncates the sweep rather than
    compressing it, keeping short runs comparable with the full one.
    """
    num_steps = int(round(duration / dt))
    time = np.arange(num_steps) * dt
    if ablation.dead_motor:
        held = np.full(num_steps, ablation.initial_angle)
        return Rollout(dt=dt, initial_angle=ablation.initial_angle, goal=held, is_chirp=False)
    sweep_rate = (CHIRP_END_FREQUENCY - CHIRP_START_FREQUENCY) / CHIRP_DURATION
    phase = 2.0 * math.pi * (CHIRP_START_FREQUENCY * time + 0.5 * sweep_rate * time**2)
    return Rollout(dt=dt, initial_angle=ablation.initial_angle, goal=CHIRP_AMPLITUDE * np.sin(phase), is_chirp=True)


def run_reference_rollout(model: Model, rollout: Rollout) -> tuple[np.ndarray, np.ndarray]:
    """Roll the goal trajectory through the reference CPU simulator.

    The trajectory is handed to :meth:`bam.simulate.Simulator.rollout_log` as a synthetic log,
    which is the entry point ``bam.fit`` uses to score a model against bench data -- so this is
    the reference pipeline, not a re-implementation of it. ``simulate_control=True`` closes the
    loop on the simulated state, matching how Isaac Lab's actuator reads the live joint.

    Args:
        model: Reference BAM model, already configured.
        rollout: The trajectory to drive.

    Returns:
        Joint positions [rad] and velocities [rad/s], each of shape ``(num_steps,)``, sampled
        *before* the step that follows them.
    """
    log = {
        "dt": rollout.dt,
        "mass": PENDULUM_MASS,
        "arm_mass": 0.0,
        "length": PENDULUM_LENGTH,
        "kp": KP_FW,
        "vin": VIN,
        "entries": [
            {"position": rollout.initial_angle, "speed": 0.0, "goal_position": float(target), "torque_enable": True}
            for target in rollout.goal
        ],
    }
    positions, velocities, _ = Simulator(model).rollout_log(log, simulate_control=True)
    return np.asarray(positions, dtype=np.float64), np.asarray(velocities, dtype=np.float64)


@dataclass(frozen=True)
class ActuatorRollout:
    """One Isaac Lab rollout, with the actuator telemetry its decomposition reads.

    Attributes:
        positions: Joint positions [rad], sampled before the step that follows them, shape
            ``(num_steps,)``.
        velocities: Joint velocities [rad/s], sampled with ``positions``, shape ``(num_steps,)``.
        efforts: Effort the actuator reported for each step [N.m], shape ``(num_steps,)``.
            **The two implementations report different quantities here.** Implementation A
            applies its stiction clip itself, so its effort is the whole joint torque;
            implementation B hands the friction to MuJoCo, so its effort is the motor torque
            alone and the friction the solver adds is not in it.
        external_torque: Load the actuator sized its friction budget with [N.m], shape
            ``(num_steps,)``, or None for implementation A, which never materializes its
            estimate outside :meth:`~isaaclab.actuators.BamActuator.compute`.
        friction_budget: Velocity-independent friction budget the actuator used [N.m], shape
            ``(num_steps,)``, or None for implementation A. On implementation B this is the
            value published into MuJoCo's ``dof_frictionloss``.
    """

    positions: np.ndarray
    velocities: np.ndarray
    efforts: np.ndarray
    external_torque: np.ndarray | None = None
    friction_budget: np.ndarray | None = None

    @property
    def state(self) -> tuple[np.ndarray, np.ndarray]:
        """The position/velocity pair the comparison against the reference is made on."""
        return self.positions, self.velocities


def _native_bam_controller():
    """Return the Warp BAM controller the Newton actuator path built, checked to be live.

    Newton owns the actuator objects, so the controller is reached through the manager's
    adapter rather than through the articulation. Both of the properties asserted here decide
    what the numbers in the report mean, so neither is assumed: ``solver_applies_friction``
    says MuJoCo owns the stiction (rather than the controller falling back to implementation
    A's torque-level clip), and a bound ``external_torque`` says the load is read from the
    solver rather than estimated.

    Returns:
        The single :class:`~isaaclab.actuators.newton.ControllerBam` of the fixture.
    """
    from isaaclab_newton.physics import NewtonManager

    from isaaclab.actuators.newton import ControllerBam

    adapter = NewtonManager._adapter  # noqa: SLF001
    _require(adapter is not None, "the Newton actuator adapter was not built; the native path did not activate")
    controllers = [
        actuator.controller for actuator in adapter.actuators if isinstance(actuator.controller, ControllerBam)
    ]
    _require(
        len(controllers) == 1,
        f"expected exactly one Newton-native BAM controller, found {len(controllers)}",
    )
    controller = controllers[0]
    _require(
        controller.solver_applies_friction,
        "the BAM controller is applying implementation A's torque-level stiction clip, not the solver-side"
        " friction constraint: the MuJoCo Warp bridge did not bind, so this run would measure implementation A.",
    )
    _require(
        controller.external_torque is not None,
        "the BAM controller is estimating the external torque instead of reading it from the solver.",
    )
    return controller


def run_isaaclab_rollout(
    impl: str, params_file: Path, parameters: dict[str, Any], rollout: Rollout, device: str
) -> ActuatorRollout:
    """Roll the goal trajectory through one of the two BAM implementations on Newton/MJWarp.

    Both implementations are driven through the *same* fixture, the same
    :class:`~isaaclab.actuators.BamActuatorCfg` and the same parameter file; the only
    difference is :attr:`~isaaclab.sim.SimulationCfg.use_newton_actuators`, which is what
    routes the configuration either to the Python-executed
    :class:`~isaaclab.actuators.BamActuator` or to the Warp
    :class:`~isaaclab.actuators.newton.ControllerBam`. That keeps the two rollouts comparable
    with each other as well as with the reference.

    Args:
        impl: ``"lab"`` for implementation A or ``"newton"`` for implementation B.
        params_file: BAM parameter file the actuator is configured from.
        parameters: The same parameters as a dictionary, used to author the pendulum's inertia.
        rollout: The trajectory to drive.
        device: Torch/simulation device.

    Returns:
        The recorded rollout. Positions and velocities are sampled before the step that
        follows them, and the effort is the one the actuator computed from that state.
    """
    # Imported here so ``--help`` and the revision guard do not pay for the simulator import.
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.sim import SimulationCfg, build_simulation_context

    native = impl == "newton"
    with tempfile.TemporaryDirectory() as scratch:
        usda_file = Path(scratch) / "bam_parity_pendulum.usda"
        usda_file.write_text(
            PENDULUM_USDA_TEMPLATE.format(mass=PENDULUM_MASS, com_z=-PENDULUM_LENGTH, inertia=parameters["armature"])
        )

        sim_cfg = SimulationCfg(
            dt=rollout.dt,
            device=device,
            gravity=(0.0, 0.0, -GRAVITY),
            # The only switch between the two implementations under test.
            use_newton_actuators=native,
            physics=NewtonCfg(
                solver_cfg=MJWarpSolverCfg(
                    njmax=20, nconmax=20, ls_iterations=20, integrator="implicitfast", impratio=1
                ),
                # One substep, so the solver advances once per control period like the reference.
                num_substeps=1,
                debug_mode=False,
                # Implementation B is a stateful Newton actuator, and this loop issues one
                # command per physics step -- a decimation of one, which the backend refuses to
                # CUDA-graph-capture because the double-buffered controller state would never
                # advance across replays. Eager execution is the documented escape and costs
                # nothing at this scale. Implementation A keeps the default.
                use_cuda_graph=not native,
            ),
        )
        with build_simulation_context(device=device, add_ground_plane=False, sim_cfg=sim_cfg) as sim:
            sim._app_control_on_stop_handle = None  # noqa: SLF001
            sim_utils.create_prim("/World/Env_0", "Xform", translation=(0.0, 0.0, 1.0))
            robot = Articulation(
                ArticulationCfg(
                    prim_path="/World/Env_[^/]*/Robot",
                    spawn=sim_utils.UsdFileCfg(usd_path=str(usda_file)),
                    init_state=ArticulationCfg.InitialStateCfg(joint_pos={"joint": rollout.initial_angle}),
                    actuators={
                        "servo": BamActuatorCfg(
                            joint_names_expr=[".*"],
                            params_file=str(params_file),
                            kp_fw=KP_FW,
                            vin=VIN,
                            dt=rollout.dt,
                        )
                    },
                )
            )
            sim.reset()
            _require(robot.is_initialized, "the pendulum articulation failed to initialize")
            controller = _native_bam_controller() if native else None

            robot.write_joint_position_to_sim_index(
                position=torch.full_like(robot.data.joint_pos.torch, rollout.initial_angle)
            )
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(robot.data.joint_vel.torch))
            robot.actuators.reset()

            positions, velocities, efforts = [], [], []
            external_torque, friction_budget = [], []
            target = torch.zeros(1, robot.num_joints, device=robot.device)
            for step_target in rollout.goal:
                positions.append(float(robot.data.joint_pos.torch[0, 0]))
                velocities.append(float(robot.data.joint_vel.torch[0, 0]))
                target.fill_(float(step_target))
                robot.actuators.target_command.set_position_index(value=target)
                robot.write_data_to_sim()
                # Implementation A computes its effort in ``write_data_to_sim``; implementation
                # B's controller runs inside the physics step, so its telemetry is only valid
                # once the step has been taken. Either way the effort recorded at index ``k``
                # is the one computed from the state recorded at index ``k``.
                if not native:
                    efforts.append(float(robot.actuators.applied_effort.torch[0, 0]))
                sim.step()
                robot.update(rollout.dt)
                if native:
                    efforts.append(float(robot.actuators.applied_effort.torch[0, 0]))
                    external_torque.append(float(controller.external_torque.numpy()[0]))
                    friction_budget.append(float(controller.friction_budget.numpy()[0]))

    def as_array(values: list[float]) -> np.ndarray | None:
        return np.asarray(values, dtype=np.float64) if values else None

    return ActuatorRollout(
        positions=np.asarray(positions, dtype=np.float64),
        velocities=np.asarray(velocities, dtype=np.float64),
        efforts=np.asarray(efforts, dtype=np.float64),
        external_torque=as_array(external_torque),
        friction_budget=as_array(friction_budget),
    )


def gravity_torque(positions: np.ndarray) -> np.ndarray:
    """Return the pendulum's gravity torque about the joint axis [N.m].

    This is ``Pendulum.compute_bias`` with ``arm_mass = 0``, i.e. the true external load both
    simulators work against.
    """
    return -PENDULUM_MASS * GRAVITY * PENDULUM_LENGTH * np.sin(positions)


def effective_inertia(rollout: Rollout, positions: np.ndarray, velocities: np.ndarray) -> float:
    """Return the joint inertia implied by the first step of a torque-free rollout [kg.m^2].

    Both integrators reach the same first velocity, ``dq_1 = tau_gravity(q_0) * dt / I``, so
    dividing that back out recovers each side's effective inertia. It is the sharpest available
    check that the two rigid-body models really are the same one: a mis-authored mass, lever arm
    or armature moves this number, while the integration rule does not.
    """
    _require(velocities[1] != 0.0, "the torque-free rollout did not move; the release angle cannot be an equilibrium")
    return float(gravity_torque(np.asarray(positions[0])) * rollout.dt / velocities[1])


def _column(values: np.ndarray) -> torch.Tensor:
    """Return ``values`` as the ``(N, 1)`` float64 tensor the math core operates on."""
    return torch.as_tensor(values, dtype=torch.float64).reshape(-1, 1)


def _replay_motor_torque(params: BamMotorParams, rollout: Rollout, trace: ActuatorRollout) -> torch.Tensor:
    """Re-derive the motor-side torque of every step from the recorded state [N.m].

    Runs the firmware and DC-motor stages of :mod:`isaaclab.actuators.bam_model` over the
    trajectory that was actually driven. This is a pure function of ``(target, q, dq)`` here
    because the harness configures nothing stateful ahead of it -- no supply sag, no gain
    scaling, no command delay -- which is what makes replaying either implementation legitimate.
    """
    target, position = _column(rollout.goal), _column(trace.positions)
    velocity = _column(trace.velocities)
    kp = torch.full_like(target, KP_FW)
    vin = torch.full_like(target, VIN)
    return compute_motor_torque(compute_duty(target, position, velocity, kp, vin, params), velocity, vin, params)


def external_torque_deviation(params_file: Path, rollout: Rollout, trace: ActuatorRollout) -> dict[str, float]:
    """Quantify implementation A's estimated-external-torque deviation, and what reaches the model.

    :meth:`~isaaclab.actuators.BamActuator._estimate_external_torque` reconstructs the load from
    the rotor momentum balance, ``armature * ddq - tau_applied_prev``; the true external torque
    on this pendulum is :func:`gravity_torque`. Their difference -- the link-side inertial term
    the actuator cannot observe -- is the first of the two deviations this harness measures.

    The raw torque residual overstates the deviation, because the estimate reaches the physics
    only through the friction budget, whose external-side coefficients are small and
    Stribeck-gated. The budget is therefore re-evaluated with the true load in place of the
    estimate, and the difference is the part that actually perturbs the model.

    Re-evaluating the budget means replaying the actuator's own pipeline over the recorded
    state, which is possible here because nothing stateful is configured (no supply sag, gain
    scaling or command delay). The replay is checked against the efforts the actuator really
    applied: if it ever stopped reproducing them, the run fails rather than reporting a
    decomposition of some other model.

    Args:
        params_file: BAM parameter file the rollout ran with.
        rollout: The trajectory that was driven.
        trace: The recorded implementation-A rollout.

    Returns:
        RMSE and peak of the torque residual and of the friction-budget error it causes [N.m],
        alongside the peak true load and the mean budget they should be read against [N.m].
    """
    params = BamMotorParams.from_json(params_file)
    velocity = _column(trace.velocities)
    motor_torque = _replay_motor_torque(params, rollout, trace)

    # Step 0 has no previous state to differentiate against; the actuator seeds its velocity
    # cache there and reports zero acceleration, so the replay only covers step 1 onwards.
    estimate = params.armature * (velocity[1:] - velocity[:-1]) / rollout.dt - _column(trace.efforts[:-1])
    truth = _column(gravity_torque(trace.positions[1:]))
    stribeck = compute_stribeck_coeff(velocity[1:], params)
    budget = compute_friction_budget(motor_torque[:-1], estimate, stribeck, params)
    replayed = apply_stiction_clip(
        motor_torque[1:], estimate, velocity[1:], budget, params.friction_viscous, rollout.dt, params.armature
    )
    replay_error = float(torch.max(torch.abs(replayed - _column(trace.efforts[1:]))))
    _require(
        replay_error <= REPLAY_TOLERANCE,
        f"replaying the actuator pipeline over the recorded rollout missed its efforts by {replay_error:.3e} N.m"
        f" (tolerance {REPLAY_TOLERANCE:.0e}); the decomposition below would not describe the model that ran.",
    )

    true_budget = compute_friction_budget(motor_torque[:-1], truth, stribeck, params)
    budget_error = (budget - true_budget).numpy()
    residual = (estimate - truth).numpy()
    return {
        "ext_torque_rmse_nm": float(np.sqrt(np.mean(residual**2))),
        "ext_torque_max_nm": float(np.max(np.abs(residual))),
        "ext_torque_peak_load_nm": float(np.max(np.abs(truth.numpy()))),
        "budget_error_rmse_nm": float(np.sqrt(np.mean(budget_error**2))),
        "budget_error_max_nm": float(np.max(np.abs(budget_error))),
        "budget_mean_nm": float(np.mean(true_budget.numpy())),
        "replay_error_nm": replay_error,
    }


def solver_torque_deviation(params_file: Path, rollout: Rollout, trace: ActuatorRollout) -> dict[str, float]:
    """Quantify what implementation B's *read* external torque costs, on the same scale as A's.

    Implementation B does not estimate the load: the MuJoCo Warp bridge hands the controller the
    generalized force the solver computed, and the controller publishes the resulting friction
    budget into ``dof_frictionloss``. Both quantities are recorded during the rollout, so the
    same two numbers implementation A is decomposed into can be measured directly rather than
    replayed -- ``ext_torque_*`` is how far the read load sits from the pendulum's true
    :func:`gravity_torque`, and ``budget_error_*`` is how much of that survives into the budget.

    What is left of the deviation is *staleness*, not error: the gather reads generalized forces
    MuJoCo evaluated at the start of the previous step, so the load is exact but one step old.
    ``ext_torque_read_error_nm`` pins the "exact" half (measured at float32 resolution) and the
    ``ext_torque_*`` statistics cost out the "one step old" half against the state the budget is
    actually applied to -- the same convention :func:`external_torque_deviation` reports A's
    estimator error in, so the two are directly comparable.

    Three tripwires keep the decomposition honest about the model that ran, mirroring
    :func:`external_torque_deviation`'s single replay:

    * the motor stage is replayed from the recorded state and must reproduce the efforts the
      Warp controller emitted -- which on this path *are* the motor torque, because the solver
      owns the friction;
    * the friction budget is recomputed from the recorded previous motor torque and the recorded
      read load, and must reproduce the budget the controller published;
    * the read load itself must equal the pendulum's closed-form load one step back.

    Together they say the Warp kernels are evaluating the shared math core on the state this
    decomposition assumes, without which the numbers below would describe some other model.

    Args:
        params_file: BAM parameter file the rollout ran with.
        rollout: The trajectory that was driven.
        trace: The recorded implementation-B rollout, including its telemetry.

    Returns:
        The same keys :func:`external_torque_deviation` returns, plus the budget replay error.
    """
    _require(
        trace.external_torque is not None and trace.friction_budget is not None,
        "the implementation-B rollout carries no controller telemetry to decompose",
    )
    params = BamMotorParams.from_json(params_file)
    velocity = _column(trace.velocities)
    motor_torque = _replay_motor_torque(params, rollout, trace)

    replay_error = float(torch.max(torch.abs(motor_torque - _column(trace.efforts))))
    _require(
        replay_error <= REPLAY_TOLERANCE,
        f"replaying the motor stage over the recorded rollout missed the controller's efforts by {replay_error:.3e}"
        f" N.m (tolerance {REPLAY_TOLERANCE:.0e}); the decomposition below would not describe the model that ran.",
    )

    # The gather runs before the actuators, and it reads generalized forces MuJoCo evaluated at
    # the *start* of the previous step, so the value recorded at step ``k`` is the exact load of
    # the state recorded at step ``k - 1``. Checking that first separates the two things this
    # channel can get wrong: whether the read is right (it is, to float32) and whether it is
    # current (it is one step old, which is what ``ext_torque_*`` below then costs out).
    read = _column(trace.external_torque[1:])
    read_error = float(torch.max(torch.abs(read - _column(gravity_torque(trace.positions[:-1])))))
    _require(
        read_error <= REPLAY_TOLERANCE,
        f"the load the controller read from the solver is off the pendulum's true load by {read_error:.3e} N.m"
        f" (tolerance {REPLAY_TOLERANCE:.0e}) even one step back, so it is not the generalized force it claims"
        " to be and the decomposition below would not describe the model that ran.",
    )

    # Step 0 is skipped: its gather precedes the first solve, exactly as A's estimator has no
    # previous state at step 0.
    truth = _column(gravity_torque(trace.positions[1:]))
    stribeck = compute_stribeck_coeff(velocity[1:], params)
    budget = compute_friction_budget(motor_torque[:-1], read, stribeck, params)
    budget_replay_error = float(torch.max(torch.abs(budget - _column(trace.friction_budget[1:]))))
    _require(
        budget_replay_error <= REPLAY_TOLERANCE,
        f"the friction budget the controller published differs from the math core's by {budget_replay_error:.3e} N.m"
        f" (tolerance {REPLAY_TOLERANCE:.0e}); the decomposition below would not describe the model that ran.",
    )

    true_budget = compute_friction_budget(motor_torque[:-1], truth, stribeck, params)
    budget_error = (budget - true_budget).numpy()
    residual = (read - truth).numpy()
    return {
        "ext_torque_rmse_nm": float(np.sqrt(np.mean(residual**2))),
        "ext_torque_max_nm": float(np.max(np.abs(residual))),
        "ext_torque_peak_load_nm": float(np.max(np.abs(truth.numpy()))),
        "budget_error_rmse_nm": float(np.sqrt(np.mean(budget_error**2))),
        "budget_error_max_nm": float(np.max(np.abs(budget_error))),
        "budget_mean_nm": float(np.mean(true_budget.numpy())),
        "replay_error_nm": replay_error,
        "budget_replay_error_nm": budget_replay_error,
        "ext_torque_read_error_nm": read_error,
    }


def compare(lab: tuple[np.ndarray, np.ndarray], reference: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    """Return the position and velocity error statistics of one rollout pair."""
    position_error = lab[0] - reference[0]
    velocity_error = lab[1] - reference[1]
    return {
        "rmse_position_deg": float(math.degrees(np.sqrt(np.mean(position_error**2)))),
        "rmse_velocity_rad_s": float(np.sqrt(np.mean(velocity_error**2))),
        "max_abs_position_error_deg": float(math.degrees(np.max(np.abs(position_error)))),
        "max_abs_velocity_error_rad_s": float(np.max(np.abs(velocity_error))),
    }


def phase_breakdown(
    rollout: Rollout, lab: tuple[np.ndarray, np.ndarray], reference: tuple[np.ndarray, np.ndarray]
) -> list[dict[str, float | None]]:
    """Split the rollout into :data:`NUM_PHASES` equal windows and compare each separately.

    The chirp's instantaneous frequency rises linearly with time, so an equal-time split is also
    a frequency split: it separates the quasi-static regime, where the friction budget decides
    the answer, from the resonant one, where the link's inertia does.
    """
    edges = np.linspace(0, rollout.num_steps, NUM_PHASES + 1).astype(int)
    phases: list[dict[str, float | None]] = []
    for start, stop in zip(edges[:-1], edges[1:]):
        window = compare((lab[0][start:stop], lab[1][start:stop]), (reference[0][start:stop], reference[1][start:stop]))
        window["start_s"] = start * rollout.dt
        window["stop_s"] = stop * rollout.dt
        window["start_hz"] = rollout.instantaneous_frequency(start)
        window["stop_hz"] = rollout.instantaneous_frequency(stop)
        phases.append(window)
    return phases


def check_no_divergence(name: str, positions: np.ndarray, velocities: np.ndarray) -> None:
    """Raise if a rollout left the range in which the comparison is meaningful."""
    _require(
        np.isfinite(positions).all() and np.isfinite(velocities).all(), f"{name} rollout produced non-finite state"
    )
    _require(
        np.max(np.abs(positions)) <= MAX_PLAUSIBLE_ANGLE,
        f"{name} rollout diverged: |q| reached {np.max(np.abs(positions)):.3f} rad",
    )
    _require(
        np.max(np.abs(velocities)) <= MAX_PLAUSIBLE_VELOCITY,
        f"{name} rollout diverged: |dq| reached {np.max(np.abs(velocities)):.3f} rad/s",
    )


def run(impl: str, ablation: Ablation, rollout: Rollout, device: str, bam_revision: str) -> dict[str, Any]:
    """Run one ablation on both simulators and collect every reported number."""
    parameters = build_parameters(ablation)
    reference_model = build_reference_model(parameters)
    reference_state = run_reference_rollout(reference_model, rollout)
    with tempfile.TemporaryDirectory() as scratch:
        # One file, read by the actuator under test and by the replay that decomposes its error.
        params_file = Path(scratch) / "bam_parity_params.json"
        params_file.write_text(json.dumps(parameters, indent=4))
        trace = run_isaaclab_rollout(impl, params_file, parameters, rollout, device)
        lab_state = trace.state

        check_no_divergence("reference", *reference_state)
        check_no_divergence(f"Isaac Lab ({impl})", *lab_state)
        decompose = solver_torque_deviation if impl == "newton" else external_torque_deviation
        deviation = decompose(params_file, rollout, trace)

    link_inertia = PENDULUM_MASS * PENDULUM_LENGTH**2
    results: dict[str, Any] = {
        "impl": impl,
        "ablation": ablation.name,
        "description": ablation.description,
        "bam_revision": bam_revision,
        "device": device,
        "model": parameters["model"],
        "dt_s": rollout.dt,
        "num_steps": rollout.num_steps,
        "link_inertia_fraction": link_inertia / (link_inertia + parameters["armature"]),
        "overall": compare(lab_state, reference_state),
        "phases": phase_breakdown(rollout, lab_state, reference_state),
    }
    results["overall"].update(deviation)

    if ablation.dead_motor:
        _require(
            np.max(np.abs(trace.efforts)) == 0.0,
            f"the {ablation.name!r} ablation is meant to produce no actuator torque, but Isaac Lab applied up to"
            f" {np.max(np.abs(trace.efforts)):.3e} N.m.",
        )
        lab_inertia = effective_inertia(rollout, *lab_state)
        reference_inertia = effective_inertia(rollout, *reference_state)
        mismatch = abs(lab_inertia - reference_inertia) / reference_inertia
        _require(
            mismatch <= INERTIA_MISMATCH_TOLERANCE,
            f"the two pendulums do not share a plant: Isaac Lab shows {lab_inertia!r} kg.m^2 and the reference"
            f" {reference_inertia!r} kg.m^2 ({mismatch:.3e} relative). Every other number would be meaningless.",
        )
        results["plant"] = {
            "lab_inertia_kg_m2": lab_inertia,
            "reference_inertia_kg_m2": reference_inertia,
            "analytic_inertia_kg_m2": link_inertia + parameters["armature"],
            "relative_mismatch": mismatch,
        }
    return results


def report(results: dict[str, Any]) -> None:
    """Print one ablation's results as a fixed-width table."""
    overall = results["overall"]
    print(f"\n=== impl '{results['impl']}', ablation '{results['ablation']}' -- {results['description']} ===")
    print(f"bam revision {results['bam_revision']}  model {results['model']}  device {results['device']}")
    print(f"{results['num_steps']} steps at dt = {results['dt_s']} s")
    print(f"  RMSE position           {overall['rmse_position_deg']:10.4f} deg")
    print(f"  RMSE velocity           {overall['rmse_velocity_rad_s']:10.4f} rad/s")
    print(f"  max |position error|    {overall['max_abs_position_error_deg']:10.4f} deg")
    print(f"  max |velocity error|    {overall['max_abs_velocity_error_rad_s']:10.4f} rad/s")
    print(f"  ext-torque RMSE         {overall['ext_torque_rmse_nm']:10.6f} N.m")
    print(f"  ext-torque max          {overall['ext_torque_max_nm']:10.6f} N.m")
    print(f"  peak true load          {overall['ext_torque_peak_load_nm']:10.6f} N.m")
    print(f"  budget error RMSE       {overall['budget_error_rmse_nm']:10.6f} N.m")
    print(f"  budget error max        {overall['budget_error_max_nm']:10.6f} N.m")
    print(f"  mean friction budget    {overall['budget_mean_nm']:10.6f} N.m")
    print(f"  pipeline replay error   {overall['replay_error_nm']:10.3e} N.m")
    if "budget_replay_error_nm" in overall:
        print(f"  budget replay error     {overall['budget_replay_error_nm']:10.3e} N.m")
        print(f"  ext-torque read error   {overall['ext_torque_read_error_nm']:10.3e} N.m")
    print(f"  link share of inertia   {results['link_inertia_fraction']:10.4f} -")
    if "plant" in results:
        plant = results["plant"]
        print(f"  inertia (Isaac Lab)     {plant['lab_inertia_kg_m2']:10.8f} kg.m^2")
        print(f"  inertia (reference)     {plant['reference_inertia_kg_m2']:10.8f} kg.m^2")
        print(f"  inertia (analytic)      {plant['analytic_inertia_kg_m2']:10.8f} kg.m^2")
        print(f"  relative mismatch       {plant['relative_mismatch']:10.3e} -")
    print(f"  {'phase':>22}  {'RMSE pos':>10}  {'max |err|':>10}  {'RMSE vel':>10}")
    for phase in results["phases"]:
        label = f"{phase['start_s']:.1f}-{phase['stop_s']:.1f}s"
        if phase["start_hz"] is not None:
            label += f" {phase['start_hz']:.2f}-{phase['stop_hz']:.2f}Hz"
        print(
            f"  {label:>22}  {phase['rmse_position_deg']:9.4f}d  {phase['max_abs_position_error_deg']:9.4f}d "
            f" {phase['rmse_velocity_rad_s']:8.4f}r/s"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--impl",
        choices=("lab", "newton"),
        default="lab",
        help="Which BAM implementation to compare against the reference: the Lab-executed"
        " actuator ('lab', implementation A) or the Newton-native Warp controller ('newton',"
        " implementation B, whose friction is applied by the MuJoCo solver).",
    )
    parser.add_argument(
        "--ablation",
        choices=sorted(ABLATIONS),
        default="full",
        help="Which friction terms are active on both sides. One ablation per invocation, so"
        " each result comes from a freshly built simulation.",
    )
    parser.add_argument("--device", default="cpu", help="Device the Isaac Lab rollout runs on.")
    parser.add_argument("--dt", type=float, default=DT, help="Control and physics timestep [s].")
    parser.add_argument(
        "--duration",
        type=float,
        default=CHIRP_DURATION,
        help="Rollout length [s]. Shorter values truncate the frequency sweep rather than compressing it.",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to dump the results to.")
    args = parser.parse_args()

    bam_revision = resolve_installed_bam_revision(allow_local_checkout=True)
    ablation = ABLATIONS[args.ablation]
    results = run(args.impl, ablation, build_rollout(ablation, args.dt, args.duration), args.device, bam_revision)
    report(results)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2))
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
