Added
^^^^^

* Added simulation-scoped canonical actuator storage, accessed through the
  :class:`~isaaclab.actuators.ActuatorCollection.ArticulationView` exposed by
  :attr:`~isaaclab.assets.Articulation.actuators`.
* Added exact-class actuator access through
  :attr:`~isaaclab.actuators.ActuatorCollection.ArticulationView.by_type` with
  compact type parameter views while preserving named group access.
* Added aggregation for compatible stateless actuator groups without changing
  their configuration or scoped parameter access.

Changed
^^^^^^^

* Changed actuator ownership to simulation-scoped canonical storage. Compatible
  exact actuator classes are aggregated automatically while each articulation
  continues to expose its logical actuator groups. Obtain the scoped view from
  :attr:`~isaaclab.assets.Articulation.actuators`, use its command and exact-type
  parameter views, and retain deprecated articulation forwarders only during
  their deprecation period.

Deprecated
^^^^^^^^^^

* Deprecated articulation-level actuator command setters and articulation-data
  command properties. Use the command view on
  :attr:`~isaaclab.assets.Articulation.actuators` instead.
* Deprecated ``computed_torque`` / ``applied_torque`` and dense actuator
  compatibility properties. Use
  :attr:`~isaaclab.actuators.ActuatorCollection.ArticulationView.computed_effort`,
  :attr:`~isaaclab.actuators.ActuatorCollection.ArticulationView.applied_effort`,
  and capable compact group or exact-type parameters instead.
* Deprecated ``write_actuator_stiffness_to_sim`` and
  ``write_actuator_damping_to_sim``. Use a capable group or exact-type
  ``set_parameter_index()`` call instead.
