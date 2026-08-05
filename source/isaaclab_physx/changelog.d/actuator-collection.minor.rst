Added
^^^^^

* Added CUDA graph replay for graphable Newton actuators running on the PhysX
  backend.

Changed
^^^^^^^

* Routed PhysX articulation actuator setup, compute, reset, and command
  submission through the scoped
  :class:`~isaaclab.actuators.ActuatorCollection.ArticulationView`.
* Prevented stateful Newton actuators from running inside caller-owned CUDA
  graph captures; let the PhysX adapter manage their alternating graphs.
