Added
^^^^^

* Added the contributed MicroDuck lift task ``IsaacContrib-Lift-Flat-MicroDuck`` under
  ``contrib/microduck/lift``, with an RSL-RL PPO configuration. A marble lies on the ground within
  reach; the robot folds down, closes its beak on it, and stands back up holding it. It runs on the
  Newton MJWarp backend and on the beak variant of the robot, whose fifteenth servo is what makes
  the grasp a grasp rather than a modelling convenience.
* Added ``lift_height`` under ``contrib/microduck/mdp``, a saturating ramp on the held object's
  height above the ground. It is **dense on purpose**: a sustained hold should be worth more than an
  instant, which paying per step gives without a second term to say so, and a bonus large enough to
  dominate a stack would pay it all in one control step -- a spike of order a thousand times the
  typical per-step reward, which is the leading suspect for the advantage variance that stopped the
  pick-and-place policy ever annealing its exploration noise. The ramp saturates, so there is no
  reward for throwing the object.
* Added ``test/test_microduck_lift_env.py``, whose reward audit is written in episode-return units
  rather than argued in prose. Lifting must beat grabbing-without-lifting, which must beat hovering,
  which must beat standing inert, and toppling onto the object must be worse than doing nothing.

Changed
^^^^^^^

* Made the drop point optional in ``update_pickplace_latch``. Leaving ``command_name`` unset
  disables the release entirely, which is what a task with nothing to place the object on wants: the
  lift task grabs and holds and never lets go. One state machine serves both tasks rather than a
  second code path that could drift from the first.
* Sized the lift task by **subtraction** from pick-and-place, which is where every failure on that
  task came from. It has no drop point, no ``carry_progress`` and no placement rewards; no walking,
  so no approach reward and no curriculum; and a five-second episode rather than the family's
  twenty, because the gesture takes about three and the longer episode left a fifteen-second dead
  zone that a trained policy filled by wandering two metres. Every failure pick-and-place produced
  was a carry-or-place failure rather than a grasp failure; the grasp itself always worked.
* Asked for 4000 training iterations rather than the family's 20000. The pick-and-place logs settle
  it: that task had solved its objective by iteration 2000 and the remaining 18000 bought almost
  nothing, and this task is strictly simpler.
