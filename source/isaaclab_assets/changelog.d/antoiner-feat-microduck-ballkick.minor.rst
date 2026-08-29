Added
^^^^^

* Added :data:`~isaaclab_assets.MICRODUCK_BALL_CFG`, the ball prop of upstream's MicroDuck ball-kick
  task: a free-floating 70 mm / 15 g hollow plastic sphere ported from
  `pollen-robotics/microduck_rl <https://github.com/pollen-robotics/microduck_rl>`_. It is the only
  non-robot asset in the MicroDuck family and the only one **authored rather than converted** --
  its upstream source is a 15-line MJCF holding a single analytic sphere, so it needs no generated
  USD and is available in a tree that has never run ``scripts/tools/convert_microduck.py``.

  Its inertia is authored explicitly as a **thin spherical shell**, ``(2/3) m r^2 = 1.225e-5``
  kg m², which is what the MJCF states. Nothing derives that from the geometry: a sphere prim
  carrying only a mass resolves to the uniform-density ``(2/5) m r^2 = 7.35e-6``, 40 % less, which
  would make the ball that much easier to spin up and change the roll-versus-slide balance of every
  kick. The tensor is derived from the configuration's own mass and radius so the three numbers
  cannot drift apart, and a configuration without an explicit mass is rejected rather than falling
  back to the solid-sphere tensor.

  The MJCF's sliding friction of 0.5 is carried across as well, together with the note that it is
  masked in every contact the ball actually makes: MuJoCo -- and Newton's MuJoCo Warp solver, which
  reproduces the rule -- mixes contact friction as the element-wise maximum of the two shapes, and
  the ground plane, the robot shells and the two soles all sit at 1.0. The torsional and rolling
  coefficients are MuJoCo's defaults, which are also Newton's shape defaults, so they are not
  restated.
* Added ``test/test_microduck_ball_asset.py``, which compares the spawned prop against its upstream
  MJCF compiled by MuJoCo: mass, radius, the hollow-shell inertia on all three axes and against the
  solid-sphere tensor it must not be, the single enabled collider, and the sliding, torsional and
  rolling friction both as authored in USD and as resolved into the Newton shape material. It skips
  when the pinned upstream MJCF cannot be fetched.
