# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate golden reference outputs for the BAM XL330 (m6) actuator parity tests.

The goldens are produced by the *reference* `BAM <https://github.com/Rhoban/bam>`_
package so that the Isaac Lab actuator port can be checked against it numerically
instead of against a re-derivation of its equations.

Reference version
-----------------
The reference is the ``mjlab_frictionloss`` branch of ``Rhoban/bam`` (NOT ``master``
and NOT the PyPI release of ``better-actuator-models``). Two commits on that branch
matter:

* ``62bd8ce12154340be97e06f7f41a0ca8f116d967`` -- the revision pinned by the
  ``microduck_rl`` consumer's ``uv.lock``. This is what the goldens are generated
  from (see :data:`BAM_COMMIT`).
* ``57d13ead53206a6bf0db3d66f86506ae8c2ce01a`` -- branch head at the time of
  writing (2026-08-28).

Between those two commits the control law (:meth:`VoltageControlledActuator.compute_control`),
the motor-torque equation (:meth:`VoltageControlledActuator.compute_torque`), the
``m6`` friction budget (``BamActuator._compute_friction_budget``) and
``params/xl330/m6.json`` are all unchanged, so the goldens below are identical for
either revision. What *did* change is the supply-voltage sag model, which the goldens
do not cover (they use a fixed ``vin``):

* at ``62bd8ce`` -- ``BamActuatorCfg.vin_drop_gain_range`` [V/Nm]::

      vin_eff = max(vin - vin_drop_gain * sum_j |tau_motor_prev,j|, vin_min)

  where ``tau_motor_prev`` is the previous step's *computed motor torque*
  (cached by the actuator, zeroed on reset).
* at branch head -- ``BamActuatorCfg.vin_drop_resistance_range`` [Ohm]::

      vin_eff = max(vin - R_drop * (sum_j |tau_actuator_prev,j| / kt), vin_min)

  where ``tau_actuator_prev`` is MuJoCo's ``data.qfrc_actuator`` from the previous
  solve. This is the same functional form with ``gain = R_drop / kt``, but the
  torque it reads differs.

Fixture layout
--------------
The ``.npz`` holds three groups of ``float64`` arrays, all of shape ``(1024,)``:

* inputs -- ``q_target`` [rad], ``q`` [rad], ``dq`` [rad/s], ``prev_tau`` [Nm]
  (previous-step motor-side torque), ``ext_tau`` [Nm] (external/gearbox torque).
* reference outputs -- ``duty`` [-], ``volts`` [V], ``motor_torque`` [Nm],
  ``frictionloss_budget`` [Nm], ``stribeck_coeff`` [-].
* scalars -- every key prefixed with ``attr_`` (e.g. ``attr_kt``, ``attr_R``,
  ``attr_max_current``, ``attr_bam_commit``), holding the motor/firmware constants and
  the fitted friction parameters needed to configure the port under test.

Which friction budget
---------------------
The goldens follow ``BamActuator._compute_friction_budget`` (the mjlab/GPU path that
the microduck consumer runs), *not* :meth:`bam.model.Model.compute_frictions` (the
numpy/CPU path). The two agree for m1--m5 but differ on the m6 quadratic term: the CPU
path additionally gates it on ``sign(ext_tau) != sign(motor_torque)`` and splits the
directional masks with strict inequalities, whereas the mjlab path applies the term
unconditionally with ``drive_mask = |motor_torque| > |ext_tau|``. Port the mjlab form.

Usage
-----
.. code-block:: bash

    uv run --with "git+https://github.com/Rhoban/bam@62bd8ce12154340be97e06f7f41a0ca8f116d967" \
        python scripts/tools/generate_bam_goldens.py

The friction budget is lifted verbatim out of the installed ``bam/mjlab.py`` source
(see :func:`load_reference_friction_budget`) rather than re-implemented here, so the
goldens cannot silently drift from the reference. ``bam.mjlab`` itself is never
imported because it pulls in ``mjlab``/``mujoco_warp``.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import types
from pathlib import Path

import numpy as np
import torch
from bam.actuator import TorchBackend, VoltageControlledActuator
from bam.model import Model, load_model

# Reference revision the goldens are generated from (Rhoban/bam @ mjlab_frictionloss).
BAM_COMMIT = "62bd8ce12154340be97e06f7f41a0ca8f116d967"

# Actuator under test: Dynamixel XL330 with the m6 (directional + quadratic) friction model.
MOTOR_NAME = "xl330"
MODEL_NAME = "m6"

# Firmware / supply configuration matching the microduck_rl consumer's nominal setup.
KP_FW = 200.0
VIN = 7.4
# Control timestep [s]. ``compute_control`` ignores it, but it is recorded so a
# consumer of the goldens calls the port with the same value.
DT = 0.005

# Sampling of the input grid.
NUM_SAMPLES = 1024
SEED = 0
Q_RANGE = (-np.pi, np.pi)  # joint angle and target [rad]
DQ_RANGE = (-20.0, 20.0)  # joint velocity [rad/s]
TAU_RANGE = (-1.5, 1.5)  # previous motor torque and external torque [Nm]

DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "source/isaaclab/test/actuators/data/bam_xl330_m6_goldens.npz"


def load_reference_friction_budget():
    """Extract ``BamActuator._compute_friction_budget`` from the installed ``bam.mjlab`` source.

    ``bam.mjlab`` imports ``mjlab`` and ``mujoco_warp``, neither of which is needed to
    evaluate the friction budget: the method only touches ``torch`` and the BAM
    :class:`~bam.model.Model` it is bound to. The function definition is therefore
    parsed out of the module source and compiled in isolation, which keeps the goldens
    tied to the reference implementation byte-for-byte.

    Returns:
        The unbound ``_compute_friction_budget`` function. Call it as
        ``fn(shim, motor_torque, external_torque, stribeck_coeff)`` where ``shim`` is any
        object exposing ``_bam_model``.
    """
    spec = importlib.util.find_spec("bam.mjlab")
    if spec is None or spec.origin is None:
        raise RuntimeError("Could not locate bam/mjlab.py in the installed bam package.")
    source_path = Path(spec.origin)
    tree = ast.parse(source_path.read_text())

    def _find(nodes, node_type, name):
        for node in nodes:
            if isinstance(node, node_type) and node.name == name:
                return node
        raise RuntimeError(f"{name} not found in {source_path}; the reference layout changed.")

    cls = _find(tree.body, ast.ClassDef, "BamActuator")
    fn = _find(cls.body, ast.FunctionDef, "_compute_friction_budget")

    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict = {"torch": torch}
    exec(compile(module, filename=str(source_path), mode="exec"), namespace)  # noqa: S102
    return namespace["_compute_friction_budget"]


def build_reference_model() -> Model:
    """Load the BAM ``xl330``/``m6`` model configured like the microduck_rl consumer."""
    model = load_model(motor_name=MOTOR_NAME, model=MODEL_NAME)
    actuator = model.actuator
    if not isinstance(actuator, VoltageControlledActuator):
        raise RuntimeError(f"Expected a VoltageControlledActuator, got {type(actuator).__name__}.")
    actuator.vin = VIN
    actuator.kp = KP_FW
    # Torch backend so the clamps vectorize over the sample batch, exactly as BamActuator does.
    actuator.backend = TorchBackend()
    return model


def sample_inputs() -> dict[str, np.ndarray]:
    """Draw the reproducible input grid. NumPy's PCG64 keeps this stable across platforms."""
    rng = np.random.default_rng(SEED)
    return {
        "q_target": rng.uniform(*Q_RANGE, NUM_SAMPLES),
        "q": rng.uniform(*Q_RANGE, NUM_SAMPLES),
        "dq": rng.uniform(*DQ_RANGE, NUM_SAMPLES),
        "prev_tau": rng.uniform(*TAU_RANGE, NUM_SAMPLES),
        "ext_tau": rng.uniform(*TAU_RANGE, NUM_SAMPLES),
    }


def compute_goldens(model: Model, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Run the reference pipeline over the sampled inputs.

    The three stages mirror ``BamActuator.compute``: firmware control law, DC-motor
    torque equation, and the velocity-independent friction budget. Unlike ``compute``,
    the motor-side and external torques fed to the budget are explicit inputs rather
    than MuJoCo state, so the goldens need no simulator.
    """
    actuator = model.actuator
    tensors = {k: torch.as_tensor(v, dtype=torch.float64) for k, v in inputs.items()}

    # 1. Firmware P-controller (+ current limiter, + PWM clamp) -> volts.
    volts = actuator.compute_control(tensors["q_target"], tensors["q"], tensors["dq"], DT)
    # ``compute_control`` returns ``vin * duty_cycle``; recover the duty it clamped to.
    duty = volts / VIN

    # 2. DC-motor equation with back-EMF -> motor torque [Nm].
    motor_torque = actuator.compute_torque(volts, True, tensors["q"], tensors["dq"])

    # 3. Stribeck coefficient (1 at rest, 0 when moving), as computed in BamActuator.compute.
    stribeck_coeff = torch.exp(-torch.pow(torch.abs(tensors["dq"]) / model.dtheta_stribeck.value, model.alpha.value))

    # 4. m6 friction budget from the reference implementation.
    friction_budget_fn = load_reference_friction_budget()
    shim = types.SimpleNamespace(_bam_model=model)
    frictionloss_budget = friction_budget_fn(shim, tensors["prev_tau"], tensors["ext_tau"], stribeck_coeff)

    return {
        "duty": duty.numpy(),
        "volts": volts.numpy(),
        "motor_torque": motor_torque.numpy(),
        "frictionloss_budget": frictionloss_budget.numpy(),
        "stribeck_coeff": stribeck_coeff.numpy(),
    }


def collect_scalars(model: Model) -> dict[str, float | str | int]:
    """Collect the model/firmware scalars a consumer needs to configure the port."""
    actuator = model.actuator
    scalars: dict[str, float | str | int] = {
        "bam_commit": BAM_COMMIT,
        "motor_name": MOTOR_NAME,
        "model_name": MODEL_NAME,
        "seed": SEED,
        "num_samples": NUM_SAMPLES,
        # Firmware / supply.
        "kp": KP_FW,
        "vin": VIN,
        "dt": DT,
        "error_gain": actuator.error_gain,
        "max_pwm": actuator.max_pwm,
        "max_current": actuator.max_current,
    }
    # Every fitted parameter of the model (kt, R, armature, friction terms, ...).
    scalars.update({name: param.value for name, param in model.get_parameters().items()})
    return scalars


def check_goldens(scalars: dict, inputs: dict[str, np.ndarray], goldens: dict[str, np.ndarray]) -> None:
    """Sanity-check the generated goldens against independently derived bounds."""
    kt = scalars["kt"]
    resistance = scalars["R"]
    max_current = scalars["max_current"]

    # Duty cycle can never leave the physical PWM range (clamped last in compute_control).
    assert np.all(np.abs(goldens["duty"]) <= scalars["max_pwm"] + 1e-12), "duty exceeds max_pwm"
    assert np.allclose(goldens["volts"], goldens["duty"] * VIN), "volts != duty * vin"

    # Motor torque is kt * I. The firmware limiter targets |I| <= max_current but can only
    # clamp the duty, so at high back-EMF the PWM rail wins: |I| <= (vin + kt*|dq|max) / R.
    max_speed = max(abs(DQ_RANGE[0]), abs(DQ_RANGE[1]))
    torque_bound = kt * max(max_current, (VIN + kt * max_speed) / resistance)
    assert np.all(np.abs(goldens["motor_torque"]) <= torque_bound + 1e-9), "motor torque exceeds the DC-motor bound"

    # Every friction term added on top of friction_base is non-negative.
    assert np.all(goldens["frictionloss_budget"] >= scalars["friction_base"] - 1e-12), "budget below friction_base"

    # Stribeck coefficient is a decaying exponential in |dq|.
    assert np.all((goldens["stribeck_coeff"] >= 0.0) & (goldens["stribeck_coeff"] <= 1.0)), "stribeck outside [0, 1]"

    # Independent recomputation of the torque equation (tau = kt*V/R - kt^2*dq/R).
    expected_torque = kt * goldens["volts"] / resistance - (kt**2) * inputs["dq"] / resistance
    assert np.allclose(goldens["motor_torque"], expected_torque), "motor torque does not match the DC-motor equation"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path of the .npz fixture to write.")
    args = parser.parse_args()

    model = build_reference_model()
    inputs = sample_inputs()
    goldens = compute_goldens(model, inputs)
    scalars = collect_scalars(model)
    check_goldens(scalars, inputs, goldens)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **inputs, **goldens, **{f"attr_{k}": np.asarray(v) for k, v in scalars.items()})

    print(f"Wrote {args.output} ({args.output.stat().st_size} bytes)")
    print(f"bam commit: {BAM_COMMIT}  motor: {MOTOR_NAME}/{MODEL_NAME}  kp={KP_FW}  vin={VIN}")
    for name, array in {**inputs, **goldens}.items():
        print(
            f"  {name:20s} shape={array.shape} min={array.min(): .6f} max={array.max(): .6f} mean={array.mean(): .6f}"
        )


if __name__ == "__main__":
    main()
