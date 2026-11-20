* Changed :func:`~isaaclab.actuators.newton.read_group_parameter` and
  :func:`~isaaclab.actuators.newton.write_group_parameter` to index the Newton actuator's parameter arrays
  directly, so a write touches only the addressed environments and joints instead of gathering and scattering
  every joint of every environment through Newton's selection API.
* **Breaking:** Changed :class:`~isaaclab.actuators.newton.adapter.NewtonActuatorSelection` to carry the DOF of
  every joint (``joint_dof_indices``) instead of ``view`` and ``joint_user_to_backend_indices``. Custom
  :meth:`~isaaclab.actuators.ActuatorControl.finalize_native_actuators` implementations should return
  ``NewtonActuatorSelection(actuators=...)``, passing ``joint_dof_indices`` unless the actuators index joints
  as ``env * num_joints + joint``.
