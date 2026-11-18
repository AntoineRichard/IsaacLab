Christmas ornament on a VBD tree
================================

A native-Newton demo in which a Unitree G1 with a fixed lower body hangs a glass ornament on one
outer branch of a simulated fir tree. The robot, the ornament and the ~2.4k-segment tree all run in
one ``SolverVBD``: there is no solver coupling, and nothing welds, attaches or kinematically drives
the ornament. It starts hanging from the G1's extended right index finger and ends on the branch,
held by contact alone; the branch dips about a centimetre under it. Like the stocking atelier, it
uses Newton's viewer directly and does not register an Isaac Lab environment.

Run from the Isaac Lab repository root in the uv environment. This helper uses the tested upstream
Newton revision in a temporary uv overlay, as the atelier does:

.. code-block:: bash

   ornament() {
       uv run --no-sync \
           --with "newton @ git+https://github.com/newton-physics/newton.git@405ec9b54c3743b126cafe3ea59416a921ac36c8" \
           --with "warp-lang==1.17.0" \
           python -m examples.christmas_ornament."$@"
   }

   # Watch the full scene in the OpenGL viewer.
   ornament run --device cuda:0

   # Qualify the hand-over on a standalone branch (about 45 s on an RTX 5090), then the full tree.
   ornament run --mode probe --viewer null --check
   ornament run --mode hang --viewer null --check --metrics /tmp/ornament.json

   # Render MP4s headless (cameras: wide, close, hand).
   PYGLET_HEADLESS=1 ornament render --cam close --out /tmp/ornament_close.mp4

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
are render-only cones (about 258k) placed each frame from the segment poses.

The tree leaves a slot toward the robot: long lower branches are phased around it and twigs that
would reach into the robot's standing sector or the approach corridor are pruned. One outer
branch at hand height is the target; it carries no pre-hung ornament.

``assets.py`` builds the 35 g ornament (a 60 mm rigid ring collided through a mesh SDF, a cap and
a hollow glass sphere) and imports ``g1_29dof_with_hand_rev_1_0.urdf`` from Newton Assets
revision ``a96f0973``. The pelvis is welded to the world (``collapse_fixed_joints`` merges it into
the world body), the legs are held by stiff PD, and only right-hand shapes collide. Robot shapes
use a 2 mm contact gap: the importer default (0.1 m) floods the ornament's per-body contact buffer
and drops real contacts.

Motion
------

``motion.py`` scripts hook-point waypoints in the target branch's frame, read from the settled
simulation after a 2 s settle. The hook point is on the back of the distal index phalanx, 17 mm
from the fingertip; only the index reaches that far, so the ring clears the other digits.

1. ``present`` and ``align``: bring the ring to the branch tip, finger aimed along the branch and
   pitched 5 degrees tip-up so the ring cannot slide off.
2. ``thread``: slide the ring over the bare tip, the twig passing under the finger midway down
   the ring.
3. ``lower``: come down until the finger is 12 mm above the twig.
4. ``tip``: pitch the finger 40 degrees tip-down about its fingertip. That is past the
   ring-on-finger friction angle, so the ring slides off the fingertip onto the twig, inboard of
   the finger.
5. ``withdraw``, ``retreat``, ``home``, then ``admire`` for 5 s while the ornament settles.

Each phase is densified into eight Cartesian samples on minimum-jerk timing and solved with IK on
a separate reference model; joints interpolate linearly between samples. Interpolating between
whole-phase IK solutions instead swung the hand off its path, pressed the twig down and catapulted
the ornament when the branch sprang back.

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
the loaded tip stays level and the hang point dips about 10 mm. Branch bend damping is 0.15 s,
since bouncing walks the ring outward.

The ring is chunkier than a real loop (60 mm) because it has to pass a twig under a 26 mm Dex3
finger, about 15 degrees off its axis.

Verified configuration
----------------------

On 2026-11-18, Newton revision ``405ec9b54c3743b126cafe3ea59416a921ac36c8`` with Warp 1.17.0 on
an RTX 5090, at the default 20 substeps and 20 VBD iterations at 60 FPS:

* ``--mode probe --check`` passed 8 of 8 consecutive runs, and 4 of 4 again with the creep,
  tip-margin and clearance gates added after review (dip 9.2 to 10.3 mm, final finger
  clearance about 16 cm, peak hook error 0.6 mm), about 45 s each;
* ``--mode hang --check`` passed 3 of 3 runs with the full tree (dip 9.9 to 10.2 mm, peak hook
  error 0.8 mm, trunk-top motion 0.2 mm, ring creep 0.5 mm, no non-target contacts), about
  5 minutes each with CUDA graph capture and per-frame measurement;
* the test suite passed (25 tests, about 3 minutes including the probe).

GPU contact ordering makes runs non-deterministic, so judge changes over several probe runs rather
than one. These runs shared the GPU with other jobs; they qualify correctness, not real-time speed.
