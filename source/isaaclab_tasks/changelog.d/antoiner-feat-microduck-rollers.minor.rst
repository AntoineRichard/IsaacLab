Added
^^^^^

* Added the contributed MicroDuck roller-skating task ``IsaacContrib-Velocity-Flat-MicroDuck-Rollers``
  under ``contrib/microduck/velocity``, with an RSL-RL PPO configuration carrying upstream's
  hyper-parameters. It is ported from
  `pollen-robotics/microduck_rl <https://github.com/pollen-robotics/microduck_rl>`_ and runs on the
  Newton MJWarp backend. The robot spawns :data:`~isaaclab_assets.MICRODUCK_ROLLERS_CFG`, which
  replaces each foot with a two-wheel bogie, so it cannot walk and can only get anywhere by pushing:
  the surge command is a **throttle** rather than a velocity target -- zero coasts, positive pushes,
  negative brakes -- and the only positive task reward is the wheel speed. It is registered as a
  variant of the velocity family, which is where upstream registers it and where its recipe comes
  from, and it shares the family's 61-wide deployed observation contract with the head-pose and
  body-pose slots zero-padded. Its four passive wheel hinges are excluded from the action space by
  the servo group's existing ``^(?!passive_).*`` selector, so the action stays 14 wide on an 18-joint
  robot; the critic gains a privileged four-wide wheel-speed observation.
* Added the MicroDuck roller-skating reward terms under ``contrib/microduck/mdp``:
  ``wheel_speed_reward``, ``braking_reward``, ``skating_air_time_reward``, ``single_support_reward``,
  ``glide_reward``, ``gait_symmetry_penalty``, ``forward_lean_reward``, ``heading_hold_reward``,
  ``feet_flat_penalty``, ``com_height_target``, ``action_over_limit_penalty``,
  ``joint_action_rate_l2`` and ``joint_pose_l2``. Six of them shape the skating stroke against the
  *swizzle*, the degenerate both-blades-down waddle a naive contact reward converges to.
  ``action_over_limit_penalty`` charges the joint *command* past its hard stop rather than the
  achieved position, which is upstream's deliberate replacement for an environment-side action clip:
  the deployed runtime does not clip, so a clip would exist only in simulation.
* Added ``RelativeHeadingVelocityCommand`` under ``contrib/microduck/mdp``, whose yaw slot carries a
  wrapped heading *error* toward a per-episode target rather than a yaw rate. On the shipped
  configuration the yaw range is ``(0.0, 0.0)``, so the error is computed and then clamped away and
  turning is disabled -- upstream focuses on straight-line skating and holds the heading with a
  reward instead. The machinery is carried across because re-enabling turning is a range change.
* Added ``randomize_joint_dry_friction`` under ``contrib/microduck/mdp``, which sets the dry friction
  of selected joints without touching their viscous term. The roller task drives it on the four
  bearings, whose MJCF friction is zero for trainability, and ramps a realistic drag in by curriculum.
  The stock :func:`~isaaclab.envs.mdp.randomize_joint_parameters` is not reusable: its
  ``friction_distribution_params`` writes the sampled value into the dry *and* the viscous
  coefficient, and the bearings have no authored viscous term, so at the top of the ramp the extra
  viscous drag would brake a wheel at skating speed an order of magnitude harder than the dry
  friction it models.
* Added ``fold_bodies_into_feet`` under ``contrib/microduck/mdp``, and a ``bodies_per_foot``
  parameter to ``foot_contact``, ``foot_air_time_safe`` and ``foot_contact_forces_safe``. Upstream
  gets the family's two-slot, left-first foot semantics from a subtree contact sensor that reduces
  each ankle's two tires to one slot; Isaac Lab's contact sensor reports one slot per body, so the
  reduction happens in the terms instead -- smallest air time, largest contact time, summed force.
  The parameter defaults to one collider per foot, so every other MicroDuck task is unaffected.
* Added ``test/test_microduck_rollers_env.py``, which compares the assembled configuration term by
  term against the upstream reward, event, curriculum, command and observation tables without
  launching the simulator, and then runs the acceptance tests for the thing that makes this task
  different: that a scripted push spins the passive wheels through ground contact alone, that the
  same push with braked bearings does not, and that the rolling robot glides further than the
  skidding one. It skips the simulator tests when the generated roller MicroDuck USD is absent.

Changed
^^^^^^^

* Reproduced two upstream numbers **verbatim although both are known to be stale**, because the
  deployed skating policies were trained against them and correcting either is a retune with its own
  training run rather than a port fix. The ``com_height_target`` band ``(0.0935, 0.1235)`` m sits
  entirely below the roller model's own standing height of 0.1407 m, so at weight 2.0 it asks for a
  permanent 1.7 to 4.7 cm crouch; it is sized against the wheel-less model this environment used to
  load by mistake. The wheel-speed reward's ``wheel_radius`` is upstream's 0.0175 m default against a
  measured tire radius of 0.0150 m, so its saturation point is a ground speed of 0.257 m/s rather
  than the intended 0.300. Both are commented where they are configured.
* Changed the roller task's NaN guarding to follow the stand-up task's rather than upstream's, as the
  forward-roll port does. Upstream leaves ``robot_state_is_nan``'s sensor list empty here and reads
  its critic contact terms unguarded, which its own extraction reads as drift and recommends closing
  on this task in particular -- the critic's wheel-speed observation is fed by free-spinning
  unlimited joints, which the NaN termination's own documentation names as an explosion source. The
  guard only changes behaviour in states that are already broken.
* The task overrides :data:`~isaaclab_assets.MICRODUCK_ROLLERS_CFG`'s spawn height, which that
  configuration's own warning requires: wheels are taller than soles, and the inherited 0.125 m would
  bury the tires 1.6 cm in the floor. It is set to 0.1385 m, the midpoint of upstream's reset band,
  so the reset event's symmetric offset reproduces that band absolutely.
* The blade-flatness reward measures the ankle **body** frame about a configured normal axis, where
  upstream measures an MJCF foot *site* Isaac Lab has no equivalent of. Both sites are rotated
  relative to their ankle body and both rotations carry the site's ``z`` axis onto the body's ``+y``
  axis, which an integration test measures on the converted asset rather than assuming.
* The self-collision sensor carries the same documented narrowing the stand-up and forward-roll ports
  do: it senses the trunk against the collider-carrying bodies below it, which reports the same
  0-or-1 signal but does not see one blade clipping the other. Widening it needs the Newton backend's
  shape-level ``sensor_shape_prim_expr`` / ``filter_shape_prim_expr`` filtering and is tracked as
  separate work.
