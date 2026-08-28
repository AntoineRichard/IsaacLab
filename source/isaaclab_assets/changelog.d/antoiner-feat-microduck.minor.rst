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
* Added the two remaining upstream MicroDuck robot models to the conversion tooling, selected with
  ``scripts/tools/convert_microduck.py --model {walk,allcollisions,rollers}`` (``walk`` stays the
  default, so the existing command is unchanged). ``allcollisions`` adds the trunk, hip, shin and
  head colliders upstream's stand-up and roulade tasks need; ``rollers`` replaces each foot with two
  passively rolling wheels. The world-contact set a conversion keeps is re-derived from that model's
  own MJCF ``contype``/``conaffinity`` rather than shared between models, so the three head shells on
  ``jaw_soft`` stay collidable -- a task that rolls the robot over its head needs a head that touches
  the ground.
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
