Changed
^^^^^^^

* Routed OVPhysX articulation actuator setup, compute, reset, and command
  submission through the scoped
  :class:`~isaaclab.actuators.ActuatorCollection.ArticulationView`. Continue to
  access actuators through :attr:`~isaaclab.assets.Articulation.actuators`; no
  backend-specific migration is required.
