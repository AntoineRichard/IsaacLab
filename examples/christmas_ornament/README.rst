Christmas ornament on a VBD tree
================================

A native-Newton demo in which an Agile One SN with Wuji hands and a fixed lower body hangs a glass
ornament on one outer branch of a simulated fir tree, moving only its right arm. The robot, the
ornament and the ~2.4k-segment tree all run in one ``SolverVBD``: there is no solver coupling, and
nothing welds, attaches or kinematically drives the ornament. It starts hanging from the robot's
extended right index finger and ends on the branch, held by contact alone; the branch dips about
1.5 cm under it. Like the stocking atelier, it uses Newton's viewer directly and does not register
an Isaac Lab environment.

The Agile One URDF is internal and has no download. Point ``AGILE_ONE_URDF`` at
``agile_robotics/agile_one_sn/generated/hands_wuji/urdf/hands_wuji.urdf`` in a
``robot_menagerie`` checkout with its Git LFS meshes pulled. Without it, the tests that need the
robot skip.

Run from the Isaac Lab repository root in the uv environment. This helper uses the tested upstream
Newton revision in a temporary uv overlay, as the atelier does:

.. code-block:: bash

   export AGILE_ONE_URDF=/path/to/robot_menagerie/agile_robotics/agile_one_sn/generated/hands_wuji/urdf/hands_wuji.urdf
   ornament() {
       uv run --no-sync \
           --with "newton @ git+https://github.com/newton-physics/newton.git@405ec9b54c3743b126cafe3ea59416a921ac36c8" \
           --with "warp-lang==1.17.0" \
           python -m examples.christmas_ornament."$@"
   }

   # Watch the full scene in the OpenGL viewer.
   ornament run --device cuda:0

   # Qualify the hand-over on a standalone branch (about 75 s on an RTX 5090), then the full tree.
   ornament run --mode probe --viewer null --check
   ornament run --mode hang --viewer null --check --metrics /tmp/ornament.json

   # Render MP4s headless (cameras: wide, close, hand) with the OpenGL viewer ...
   PYGLET_HEADLESS=1 ornament render --cam close --out /tmp/ornament_close.mp4

   # ... or path-traced with ViewerRTX. Newton at this revision needs ovrtx >= 0.5.1; the Isaac Lab
   # environment ships 0.4.1, which segfaults during scene population, so add it to the overlay.
   PYGLET_HEADLESS=1 uv run --no-sync \
       --with "newton @ git+https://github.com/newton-physics/newton.git@405ec9b54c3743b126cafe3ea59416a921ac36c8" \
       --with "warp-lang==1.17.0" --with "ovrtx>=0.5.1.385782" --with "ovstage>=0.2.1.385922" \
       python -m examples.christmas_ornament.render --renderer rtx --environment studio --cam close \
       --out /tmp/ornament_close_rtx.mp4

   uv run python -m pytest examples/christmas_ornament/tests -q

``--check`` exits non-zero unless every gate below passes. ``--num-frames`` truncates the 21.5 s
sequence; ``--wind`` adds the gusting wind from the tree prototype (off by default).

Scene
-----

``tree.py`` builds the fir from capsule segments joined by VBD rod joints. Bend stiffness follows
beam theory, ``E (pi r^4 / 4) / span``, with two moduli: the trunk at 5 GPa so it stands, and
branches at 0.2 GPa. Branch rod stretch stiffness is 1e5 N/m rather than the trunk's 1e7 N/m:
at 1e7 against bend stiffness of a few N m/rad, 20 VBD iterations left branch bending about ten
times under-converged, and the tree drooped from solver softness rather than material. Needles
are render-only cones (about 258k) placed each frame from the segment poses. ``ViewerRTX`` builds a
USD prim per instance, which takes many minutes for that many needles, so ``--renderer rtx`` bakes
each segment's needles into one visual-only mesh shape (no collision, no mass) on that segment.

The tree leaves a slot toward the robot: long lower branches are phased around it and twigs that
would reach into the robot's standing sector or the approach corridor are pruned. One outer
branch with its hang point at about 0.93 m is the target; it carries no pre-hung ornament.

``assets.py`` builds the 35 g ornament (a 60 mm rigid ring collided through a mesh SDF, a cap and
a hollow glass sphere) and imports the Agile One. The pelvis is welded to the world
(``collapse_fixed_joints`` merges it into the world body), the legs are held by stiff PD, the
idle left arm hangs at the robot's side, and only right-hand shapes collide. Robot shapes use a
2 mm contact gap: the importer default (0.1 m) floods the ornament's per-body contact buffer and
drops real contacts.

Motion
------

``motion.py`` scripts hook-point waypoints in the target branch's frame, read from the settled
simulation after a 2 s settle. Only the seven right-arm joints move; the waist and head hold their
zero pose. The hand points forward thumb-up, as it rests, and the ring rides on the thumb side of
the distal index phalanx, 16 mm from the fingertip; only the index reaches that far, so the ring
clears the other, curled digits.

1. ``present`` and ``align``: bring the ring to the branch tip, finger aimed along the branch and
   pitched 5 degrees tip-up so the ring cannot slide off.
2. ``thread``: slide the ring over the bare tip, the twig passing beside the finger, 4 mm clear
   of it, and midway down the ring.
3. ``lower``: come straight down past the twig until the finger sits at the centre of the ring's
   hole as the ring will hang from the twig. The ring lands on the twig on the way.
4. ``release``: hold still for 1 s while the ring settles.
5. ``withdraw``: slide the finger back out along its own axis, touching neither ring nor twig.
6. ``retreat``, ``home``, then ``admire`` for 5 s while the ornament settles.

Each phase is densified into eight Cartesian samples on minimum-jerk timing and solved with IK on
a separate reference model; joints interpolate linearly between samples. Each sample starts from
the previous solution. Where that stalls, IK re-solves from 128 random arm seeds and keeps the
solution nearest the previous one, and planning fails rather than accept an arm-joint jump above
0.5 rad (a switch to another elbow or wrist branch, which would swing the hand off its path). The
rest pose is also solved from random seeds, keeping the one farthest from the joint limits.
Interpolating between whole-phase IK solutions instead swung the hand off its path, pressed the
twig down and catapulted the ornament when the branch sprang back.

Checks
------

``--check`` requires, for the full sequence:

* finite state and a hook-point tracking error below 25 mm;
* the ring encircling the target twig on every frame of ``admire``;
* finger-to-ring clearance above 5 mm from the end of ``withdraw`` on, where a dragging finger
  would still be next to the ring;
* the ring drifting less than 2 mm along the twig over the last 2 s of ``admire`` (settling after
  the drop is allowed), with at least 10 mm of twig left before the segment's end;
* the hang point dipping at least 3 mm under the ornament;
* the ornament nearly at rest at the end (below 0.05 m/s);
* in ``hang`` mode, the trunk top moving less than 5 cm.

The JSON metrics also report ``nontarget_contacts``, the number of frames in which the ornament
touches tree parts other than the target branch, and carry per-phase samples (ring centre, hang point, hook error) that show
where a failed run lost the ornament.

Design notes
------------

The branch modulus trades the dip against the hold. Converged, a 0.9 GPa branch dips about 1.5 mm
under 35 g. Much softer than 0.2 GPa, the outermost twig joint tips the loaded segment downhill
(from +3 to -16 degrees at 0.05 GPa) and the ring creeps off the end with each bounce. At 0.2 GPa
the loaded tip stays level. These were measured on a shorter branch two whorls up; the current,
longer target branch dips about 15 mm. Branch bend damping is 0.15 s, since bouncing walks the ring outward.

The arm-only motion fixes the layout. The choreography plans in a *hook frame* (+X down the
finger, +Z out of the surface the ring rests on); ``assets.HOOK_FRAME`` maps the Wuji distal link
to it. Mapping the right arm's reach with the waist locked, from 64 random seeds per point:

* with the nail up, the finger cannot be held level in front of the chest at all (the forearm
  would have to roll past its limit), so the ring rides on the thumb side instead;
* thumb-up, the finger holds level up to about 1 m in front of the robot, so the target is the
  whorl whose hang point is at 0.93 m and the rest pose is low and forward;
* tipping the finger 40 degrees down, which an earlier hand-over used to slide the ring off the
  fingertip, needs sideways wrist bend this arm barely has; lowering the finger past the twig
  needs none.

The robot stands 0.10 m back from the tree's nominal robot position, so the hang point is 0.50 m
ahead and 0.08 m to the right of its pelvis. The ring stays 60 mm across: the finger is 16 mm,
and it has to pass beside a twig inside the ring with clearance on both sides.

Verified configuration
----------------------

On 2026-11-19, Newton revision ``405ec9b54c3743b126cafe3ea59416a921ac36c8`` with Warp 1.17.0 on
an RTX 5090, at the default 20 substeps and 20 VBD iterations at 60 FPS, with the Agile One from
``robot_menagerie`` commit ``aad8d176`` (``hands_wuji``):

* ``--mode probe --check`` passed (dip 15.2 mm, peak hook error 1.6 mm, ring creep 0.03 mm,
  final finger clearance 80 mm), about 75 s;
* ``--mode hang --check`` passed with the full tree (dip 15.3 mm, peak hook error 1.6 mm,
  trunk-top motion 0.2 mm, ring creep 0.2 mm, no non-target contacts), about 5 minutes with CUDA
  graph capture and per-frame measurement;
* the fast tests passed (29 tests, about 50 s);
* ``ViewerRTX`` renders (ovrtx 0.6.0, ovstage 0.3.0, 1280 x 960) took about 4 minutes per
  camera for the whole sequence, including about a minute of scene setup.

GPU contact ordering makes runs non-deterministic, so judge changes over several probe runs rather
than one. These runs shared the GPU with other jobs; they qualify correctness, not real-time speed.
