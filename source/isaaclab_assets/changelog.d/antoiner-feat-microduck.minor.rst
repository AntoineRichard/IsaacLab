Added
^^^^^

* Added MicroDuck walk asset conversion tooling and its fidelity tests under
  ``data/Robots/PollenRobotics/MicroDuck`` and ``test/test_microduck_asset.py``. The USD itself is
  generated on demand by ``scripts/tools/convert_microduck.py`` rather than committed, because USD
  files are excluded from the repository; the script fetches the Apache-2.0 licensed
  ``pollen-robotics/microduck_rl`` MJCF from a pinned commit, so no manual checkout is needed. See
  the ``ATTRIBUTION.md`` next to the asset for provenance and the conversion command.
* Added a conversion step that clears the articulation root transform the MJCF importer bakes the
  model's home pose into, so an ``ArticulationCfg``'s initial position sets the spawn height instead
  of composing with it and doubling it.
* Added tests comparing the generated USD against the source MJCF: joint names, position limits,
  armature, effort limits, body masses, root transform, world-contact colliders and foot friction.
  The actuator gains, joint damping and joint friction are not carried by the conversion and must be
  supplied by the task's actuator configuration; the tests pin that too.
* Added :data:`~isaaclab_assets.MICRODUCK_CFG`, the MicroDuck articulation in the upstream stand
  pose, driven by :class:`~isaaclab.actuators.BamActuatorCfg` at upstream's deployment settings: the
  vendored Dynamixel XL330 ``m6`` fit, a firmware gain of 200, a per-robot battery voltage, sag and
  gearbox-friction draw, and a 3 to 6 physics-step command delay. It restores the joint damping and
  friction the conversion drops and bounds the model at the electrical stall torque upstream derives
  from the top of its battery range. Spawning it without the generated USD raises an error naming
  the command that regenerates it.
