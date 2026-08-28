Added
^^^^^

* Added the backend-agnostic math core of the BAM voltage-domain servo model:
  :class:`~isaaclab.actuators.BamMotorParams` and the stateless functions
  :func:`~isaaclab.actuators.compute_duty`, :func:`~isaaclab.actuators.compute_motor_torque`,
  :func:`~isaaclab.actuators.compute_stribeck_coeff`, :func:`~isaaclab.actuators.compute_friction_budget`,
  :func:`~isaaclab.actuators.apply_stiction_clip` and :func:`~isaaclab.actuators.battery_sag`.
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
  ranges once the native actuators are finalized. The command delay is owned by the controller
  because Newton's delay component has static lags. On a solver that can apply joint dry
  friction, the controller publishes its friction budget instead of clipping the torque itself.
