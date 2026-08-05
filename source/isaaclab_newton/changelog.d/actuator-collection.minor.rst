Added
^^^^^

* Added explicit state-buffer advancement so Newton actuator adapters can be
  replayed from backend-owned CUDA graphs.

Changed
^^^^^^^

* Routed Newton articulation actuator setup, compute, reset, and command
  submission through the scoped
  :class:`~isaaclab.actuators.ActuatorCollection.ArticulationView`. Continue to
  access actuators through :attr:`~isaaclab.assets.Articulation.actuators`; no
  backend-specific migration is required.

Deprecated
^^^^^^^^^^

* Deprecated :func:`~isaaclab_newton.actuators.build_newton_actuator_defaults`.
  Collection-managed integrations now bind canonical controller storage; use
  the scoped actuator facade exposed by :class:`~isaaclab.assets.Articulation`
  instead of creating a separate Newton gain snapshot.
