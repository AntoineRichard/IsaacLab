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
