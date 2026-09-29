Added
^^^^^

* Added self-contained MicroDuck USD exports under ``data/Robots/PollenRobotics/MicroDuck``
  through Git LFS, with source attribution, Apache-2.0 license and checksum manifest. Run
  ``git lfs pull`` after cloning; no MJCF download or conversion is required.
* Added checks for self-contained USDs and optional comparisons against explicitly supplied
  local MJCF references, covering joint properties, body masses and contact geometry.
* Added :data:`~isaaclab_assets.MICRODUCK_CFG`, the MicroDuck articulation in the upstream stand
  pose, driven by :class:`~isaaclab.actuators.BamActuatorCfg` at upstream's deployment settings: the
  vendored Dynamixel XL330 ``m6`` fit, a firmware gain of 200, a per-robot battery voltage, sag and
  gearbox-friction draw, and a 3 to 6 physics-step command delay. It restores the joint damping and
  friction the conversion drops and bounds the model at the electrical stall torque upstream derives
  from the top of its battery range. It loaded the exported walking USD directly through the standard USD spawner.
* Added the all-collisions and rollers USD exports with their distinct contact geometry.
  The all-collisions model retained collidable head shells; the rollers model retained four
  passive wheel hinges alongside the fourteen driven servos.
* Added :data:`~isaaclab_assets.MICRODUCK_ALLCOLLISIONS_CFG` and
  :data:`~isaaclab_assets.MICRODUCK_ROLLERS_CFG`, which are :data:`~isaaclab_assets.MICRODUCK_CFG`
  spawning those two assets. They reuse its servo group unchanged, so the roller model's four
  ``passive_*_wheel`` hinges are undriven and the action space stays 14-dimensional on all three
  robots.
* Added ``test/test_microduck_variant_assets.py``, comparing both new assets against their source
  MJCF: joint inventory including the passive wheels, per-side position limits, armature, effort
  limits, body masses, the world-contact collider set and its friction, the cleared root transform,
  and that each configuration spawns on Newton and drives exactly the 14 servos.

Changed
^^^^^^^

* Changed the passive joint damping :data:`~isaaclab_assets.MICRODUCK_CFG` restores from the MJCF's
  ``0.053`` to ``0.00536`` N·m·s/rad, the ``friction_viscous`` of the vendored Dynamixel XL330
  ``m6`` fit that upstream's servo binding publishes into MuJoCo's ``dof_damping`` -- so MicroDuck
  now integrates at the joint damping the deployed robot is identified and trained against. The
  ten-times-inflated MJCF value had only masked the underdamped joint-limit conversion on the
  MuJoCo Warp backend, which the **Breaking:** ``isaaclab_newton`` entry on unauthored joint-limit
  ``solref`` fixes. The corrected value therefore depends on that fix:
  :attr:`~isaaclab_newton.physics.MJWarpSolverCfg.use_mujoco_default_joint_limit_solref` must stay
  at its default of ``True``, since turning it off without restoring ``0.053`` here drives the
  robot to a non-finite state within a few hundred steps. Policies trained against the previous
  damping should be retrained.
