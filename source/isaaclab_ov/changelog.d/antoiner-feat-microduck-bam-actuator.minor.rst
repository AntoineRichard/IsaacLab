Changed
^^^^^^^

* A :class:`~isaaclab.actuators.BamActuatorCfg` group combined with
  :attr:`~isaaclab.sim.SimulationCfg.use_newton_actuators` now raises on this backend instead of
  being accepted. OVPhysX steps native actuators through the shared host adapter, which has no joint
  dry-friction channel for the controller to publish its friction budget into and no generalized
  forces for it to read the external load from, so the model silently degraded to the Isaac
  Lab-executed friction clip *and* never drew its start-up randomization ranges. The error names
  the fix: set ``use_newton_actuators=False`` to run
  :class:`~isaaclab.actuators.BamActuator` on this backend.
