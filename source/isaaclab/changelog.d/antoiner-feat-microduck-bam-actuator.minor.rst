Added
^^^^^

* Added the backend-agnostic math core of the BAM voltage-domain servo model:
  :class:`~isaaclab.actuators.BamMotorParams` and, in :mod:`isaaclab.actuators.bam_model`, the
  stateless functions :func:`~isaaclab.actuators.bam_model.compute_duty`,
  :func:`~isaaclab.actuators.bam_model.compute_motor_torque`,
  :func:`~isaaclab.actuators.bam_model.compute_stribeck_coeff`,
  :func:`~isaaclab.actuators.bam_model.compute_friction_budget`,
  :func:`~isaaclab.actuators.bam_model.apply_stiction_clip` and
  :func:`~isaaclab.actuators.bam_model.battery_sag`. They are reached through that module rather
  than the package root, which re-exports only the configuration-facing
  :class:`~isaaclab.actuators.BamMotorParams` and
  :data:`~isaaclab.actuators.BAM_XL330_M6_PARAMS_FILE`.
  The identified parameters of the Dynamixel XL330 are vendored in
  ``isaaclab/actuators/data/bam_xl330_m6.json``, and the port is checked against reference
  outputs generated from the upstream BAM package.
* Added :class:`~isaaclab.actuators.BamActuator` and
  :class:`~isaaclab.actuators.BamActuatorCfg`, an explicit actuator model that runs the BAM
  servo pipeline from Isaac Lab and returns a pure effort command. It supports per-environment
  supply-voltage, load-sag, friction-budget and firmware-gain randomization, and a command
  delay with a configurable resample period and hold probability. Because the actuator
  interface exposes no generalized forces, the load-dependent friction terms use an
  external-torque estimate derived from the rotor's momentum balance; see the module
  documentation for what that approximates. The inherited
  :attr:`~isaaclab.actuators.ActuatorBaseCfg.stiffness` and
  :attr:`~isaaclab.actuators.ActuatorBaseCfg.damping` fields are unused by the model and
  default to None, so an articulation configuration validates without setting them; the
  position loop is parameterized by :attr:`~isaaclab.actuators.BamActuatorCfg.kp_fw` instead.
  The model is validated end to end on a Newton MJWarp articulation, where a settling
  pendulum comes to rest inside the stiction band its own equations predict.
* Added :class:`~isaaclab.actuators.newton.ControllerBam`, the Newton-native counterpart of
  :class:`~isaaclab.actuators.BamActuator`: the same BAM pipeline as Warp kernels, executed on
  the Newton actuator fast path. It is selected by the *same*
  :class:`~isaaclab.actuators.BamActuatorCfg` when
  :attr:`~isaaclab.sim.SimulationCfg.use_newton_actuators` is enabled, which now authors a
  ``NewtonBamControlAPI`` actuator prim per driven joint instead of rejecting the config. The
  identified constants are read from the config's parameter file, so both implementations run
  the same numbers; per-environment ``vin``, ``sag_gain``, ``friction_scale``, ``kp_scale`` and
  ``kd_scale`` are reachable through
  :func:`~isaaclab.actuators.newton.write_group_parameter`, and
  :func:`~isaaclab.actuators.newton.apply_bam_startup_sampling` draws the config's start-up
  ranges once the actuator exists, since a USD prim is shared by every clone. The command delay
  is owned by the controller because Newton's delay component has static lags. On a solver that
  can apply joint dry friction, the controller publishes its friction budget instead of clipping
  the torque itself, and the new
  :attr:`~isaaclab.actuators.BamActuatorCfg.stiff_frictionloss` field stiffens that constraint;
  authoring also seeds a positive friction on the driven joints so the solver sizes a friction
  row for them from the first step. The effort limit is applied by the controller rather than by
  a composed clamping component, because Newton resolves an actuator prim's components from
  USD's *composed* schema list and would drop the unregistered ``NewtonBamControlAPI`` token as
  soon as a registered one were authored beside it.
  Because the solver owns the friction on that path,
  :attr:`~isaaclab.actuators.ActuatorCollection.applied_effort` reports the **motor** torque
  rather than the whole joint torque, and ``data.joint_friction`` reports the value authoring
  seeded rather than the live budget; read the live budget with
  :func:`~isaaclab.actuators.newton.read_group_parameter`. For the same reason the native path
  requires the Newton backend; a backend that steps native actuators through the shared host
  adapter (PhysX, OVPhysX) rejects the configuration and points at the Isaac Lab-executed model,
  which runs everywhere. Both execution paths, the vendored parameters' provenance, the
  randomization hooks and the measured parity against the upstream reference simulator are
  documented in the actuator concepts page.
* Added the :attr:`~isaaclab.actuators.ActuatorBase.applies_joint_friction` class flag. An Isaac
  Lab-executed group whose model sets it resolves its joints to zero solver static and dynamic
  friction, the way an explicit model resolves them to zero solver stiffness and damping, and
  warns when :attr:`~isaaclab.actuators.ActuatorBaseCfg.friction` or
  :attr:`~isaaclab.actuators.ActuatorBaseCfg.dynamic_friction` is configured on it.
  :class:`~isaaclab.actuators.BamActuator` sets the flag: it clips the torque against its own
  friction budget, so authored joint friction on the same joints was applied a second time by the
  solver. On the MicroDuck reference asset that double count was 0.0048 N·m per joint, and
  removing it moves the Isaac Lab-executed path onto the Newton-native path's trajectory (hold
  regime, joint RMSE against the upstream reference at 55 control steps: 0.0067 -> 0.0083, which
  is the Newton-native path's 0.0091 to within the two paths' known residual -- the lower number
  was the extra friction flattering the comparison, not better fidelity). The Newton-native path
  is unaffected: its controller republishes the budget into the solver on every physics step and
  its configured friction stays a seed. Joint viscous friction is left to the solver on both
  paths, because the reference implementation also damps the joint through MuJoCo's
  ``dof_damping``.
