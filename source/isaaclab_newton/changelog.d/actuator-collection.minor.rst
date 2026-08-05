Added
^^^^^

* Added explicit state-buffer advancement so Newton actuator adapters can be
  replayed from backend-owned CUDA graphs.

Changed
^^^^^^^

* Routed Newton articulation actuator setup, compute, reset, and command
  submission through :class:`~isaaclab.actuators.ActuatorCollection`.

Deprecated
^^^^^^^^^^

* Deprecated :func:`~isaaclab_newton.actuators.build_newton_actuator_defaults`.
  Actuator defaults are now owned by
  :class:`~isaaclab.actuators.ActuatorCollection`; use the scoped actuator
  facade exposed by :class:`~isaaclab.assets.Articulation` instead of creating
  a separate Newton gain snapshot.
