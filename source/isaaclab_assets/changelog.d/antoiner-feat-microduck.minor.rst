Added
^^^^^

* Added MicroDuck walk asset conversion tooling and its fidelity tests under
  ``data/Robots/PollenRobotics/MicroDuck`` and ``test/test_microduck_asset.py``. The USD itself is
  generated on demand by ``scripts/tools/convert_microduck.py`` rather than committed, because USD
  files are excluded from the repository; the script fetches the Apache-2.0 licensed
  ``pollen-robotics/microduck_rl`` MJCF from a pinned commit, so no manual checkout is needed. See
  the ``ATTRIBUTION.md`` next to the asset for provenance and the conversion command.
* Added tests comparing the generated USD against the source MJCF: joint names, position limits,
  armature, effort limits, body masses, root spawn height, world-contact colliders and foot
  friction. The actuator gains, joint damping and joint friction are not carried by the conversion
  and must be supplied by the task's actuator configuration; the tests pin that too.
