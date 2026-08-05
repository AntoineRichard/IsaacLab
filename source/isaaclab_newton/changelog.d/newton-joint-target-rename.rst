Changed
^^^^^^^

* **Breaking:** Renamed the Newton ``Control`` bindings from ``joint_target_pos`` /
  ``joint_target_vel`` to ``joint_target_q`` / ``joint_target_qd``, following their removal
  in Newton 1.5. ``PhysxActuatorWrapper`` mirrors the same names so an actuator can accept a
  Newton ``Control`` or the PhysX wrapper interchangeably. Update any code reading those
  attributes; the layout is unchanged while ``newton.use_coord_layout_targets`` stays at its
  default.
