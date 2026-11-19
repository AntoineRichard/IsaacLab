Christmas stocking filling atelier
==================================

A native-Newton prototype for a scripted Franka FR3 filling a supported Christmas stocking.
The robot, wrapped candies, cloth, and contacts use one ``SolverVBD``. There is no MuJoCo
coupling, kinematic grasp, or candy attachment. This prototype uses Newton's viewer directly;
it does not yet register an Isaac Lab environment or use the Isaac Lab visualizer selector.

Run from the Isaac Lab repository root in the uv environment. This shell helper uses the tested
upstream Newton revision in a temporary uv overlay; it leaves the repository dependency pin and
the installed environment unchanged:

.. code-block:: bash

   atelier() {
       uv run --no-sync \
           --with "newton @ git+https://github.com/newton-physics/newton.git@405ec9b54c3743b126cafe3ea59416a921ac36c8" \
           --with "warp-lang==1.17.0" \
           python -m examples.christmas_atelier.run "$@"
   }

   # Show the validated three-candy sequence with the sewn stocking.
   atelier --stocking-pattern sewn --substeps 12 --iterations 15 --device cuda:0

   # First qualify the articulation and physical grasp without cloth.
   atelier --mode probe --device cuda:0 --viewer null --check

   # Calibrate drape and rigid/cloth contact separately, without importing the robot.
   atelier --mode drape --device cpu --viewer null --check
   atelier --mode drop --candies 1 --device cuda:0 --viewer null --check

   # Record animated USD and a separate measurement record.
   atelier --stocking-pattern sewn --substeps 12 --iterations 15 --device cuda:0 --viewer usd \
       --output /tmp/atelier.usd --metrics /tmp/atelier.json --check

   # Export reusable, meter-scale USD art meshes without launching physics.
   atelier --stocking-pattern sewn --export-assets /tmp/christmas-assets

   uv run python -m pytest examples/christmas_atelier/tests -q

Zero ``--num-frames`` runs the entire sequence; a positive value truncates it. ``--check`` rejects
incomplete sequences. The default viewer is interactive OpenGL; ``null`` runs without a display,
and ``usd`` writes an animation. Measurements default to ``/tmp/christmas_atelier.json``.
For explicit CPU fallback, pass ``--device cpu``; this does not establish CUDA performance.
The recorded USD uses the simulation's face colors. The knit material and lighting in the
showcase video were added in Blender.

Assets and scene
----------------

``assets.py`` is the asset source. The stocking is a continuous triangulated shell with a
closed toe, shaped heel, red body, and cream cuff. The top ring is pinned to a visible brass
support. Bending edges span shared vertices across the entire shell; the cuff is the only
open boundary. Its opening is approximately 150 by 117 mm, and its height is about 350 mm.
The supported cuff is included as a point subset in the exported USD.

Each 20 g candy uses a 28 mm diameter capsule with a 10 mm straight section by default.
Select ``--candy-collider cylinder`` to try a cylinder 28 mm in diameter and 28 mm long.
The striped wrapper and twisted tails form a closed mesh collider for cloth contacts. The core handles
rigid contacts with the gripper and bin, so decorative folds
cannot snag the fingers. The two colored wrapper meshes only supply rendering. Rigid/soft
full-surface contacts (including cloth faces) are enabled in the collision pipeline.
The bin initially presents a spaced row of candies. Scripted grasping assumes these known
positions; this is the repeatable demonstration layout, rather than an arbitrary pile picker.
``--candies`` accepts one to five pieces. The robot's meshes are downloaded from Newton's
Franka asset bundle on the first robot run; use ``--robot-urdf`` for an offline FR3/Franka-hand
URDF with its mesh references available.

The arm stands on a walnut bench, with the green candy bin on its left and the stocking on
its right. A brass post supports an elliptical cuff rim; the cloth below it remains dynamic.
Candies fall freely into the sock after the fingers open. The cuff is 1.20 m above ground,
100 mm higher than the initial prototype, leaving room for the softened toe to hang freely.

Motion and solver investigation
-------------------------------

The script solves named TCP waypoints on a separate Newton IK model and interpolates joint
targets with minimum-jerk segments. VBD integrates the actual arm under PD drives. Forward
kinematics initializes the dynamics model before constructing VBD, which captures joint rest
transforms. During the rollout FK operates only on the reference model, never on dynamic poses.

URDF massless fixed frames are collapsed into physical links; the massive hand is retained,
and the TCP is measured at its authored 103.4 mm offset. Both finger joints receive explicit
targets, including on older VBD revisions without mimic constraint support. Robot self-collision
is disabled for this prototype, and arm mesh/cloth contacts are disabled; the scripted path
keeps the arm above the sock, with finger/cloth contact enabled. Arm drive gains are adjustable
with ``--arm-stiffness`` and ``--arm-damping``. This is not a hardware torque-limit validation.

The measurements include the imported Newton source path and Git revision when available. The
reported package version comes from installed distribution metadata and can differ when using a
source override through ``PYTHONPATH``. For a source checkout comparison, use:

.. code-block:: bash

   git clone https://github.com/newton-physics/newton.git /tmp/newton-atelier
   PYTHONPATH=/tmp/newton-atelier uv run --no-sync python -m examples.christmas_atelier.run \
       --mode probe --device cuda:0 --viewer null --check
   git -C /tmp/newton-atelier rev-parse HEAD

Calibration and acceptance
--------------------------

Start with ``drape`` (empty stocking), then ``drop`` (free candies above the opening), and finally
``fill``. The initial material settings are starting values; numerical stability alone does not
establish a match to real fabric. Compare the empty drape, fold scale, and loaded stretch with
a physical stocking of similar dimensions before declaring the fabric calibrated.

* ``--areal-density``: kg/m²; changes stocking weight and inertia.
* ``--stretch-stiffness``: triangle shear and area stiffness, both initially 1500.
* ``--stretch-damping``: membrane damping, initially 0.01; separate from bending damping.
* ``--bend-stiffness``: shell bending stiffness, initially 0.0000005.
* ``--bend-damping``: shell bending damping, initially 0.004.
* ``--cuff-height``: supported cuff height above ground, initially 1.20 m.
* ``--friction``: cloth contact friction, initially 0.5.
* ``--particle-radius``: contact shell radius in meters, initially 0.002.
* ``--contact-stiffness``: rigid/cloth contact stiffness, initially 2000.
* ``--mesh-rings`` and ``--mesh-sides``: simulation resolution, initially 28 by 32.
* ``--substeps`` defaults to 12. ``--iterations`` defaults to 15 for the sewn-stocking drop and 60 otherwise.
* ``--grasp-iterations``: iteration cap during pickup, carry, and early finger opening, initially 15.

Change one material parameter at a time and compare runs with the same mesh and timestep.
Check timestep and mesh convergence before interpreting softness changes as material effects.
Color balancing is disabled because Warp's balancing pass stalls on the closed-toe graph on CPU;
the graph is still colored, including bending dependencies.

A complete arm sequence passes only if every candy lifts at least 120 mm, reaches within 35 mm
of the opening during release, and peak TCP tracking error stays below 25 mm. The cloth checks
require finite positions, cuff drift below 10 µm, maximum final mesh-edge stretch below 1.35,
RMS particle speed below 30 mm/s over the final second (60 frame samples),
at least 10 mm of final bench clearance, and containment of every candy in the deformed stocking.
Core containment uses an oriented
clearance check for the selected collider against the live, temporarily capped shell.
A separate wrapper check tests vertices, edge midpoints, and triangle
centroids against the live shell, including the protruding tails.
Probe mode instead checks separation from the hand after release.
The light rigid candy uses compliant contacts with stiffness 10000 N/m and damping 30 N·s/m
by default. Use ``--grip-contact-damping`` to set the material damping of the fingertips and
candy core. Newton averages the two materials for each contact, including core contact with the
tray. A much stiffer, underdamped contact can eject the candy during closing even when arm
tracking is accurate. The original
grasp setup used 720 Hz physics stepping and 15 solve iterations. The flattened cloth uses
60 iterations for settling. Reducing the timestep resolution and iteration count together
can lose the grasp. Use separate probes for material and solver-resolution changes.

JSON contains phase samples to help distinguish missed grasps, slipping, and release errors.
These checks qualify this scene and trajectory; they do not establish general articulation
support or a visually realistic fabric response.

Initial configuration validation
--------------------------------

On 2026-11-17, Newton revision ``405ec9b54c3743b126cafe3ea59416a921ac36c8`` with Warp 1.17.0
completed the initial, stiffer three-candy sequence on an RTX 5090. Both normal stepping and CUDA graph
capture passed: all three cores remained contained after the 44.7 s sequence. The captured run's
peak TCP error was 4.99 mm, cuff drift was zero, and maximum final cloth-edge stretch was 1.037.
The separate CPU arm probe also passed. The original seven geometry, choreography, containment, and
headless export tests passed in both the existing environment and the latest-Newton overlay.

CUDA graph capture warms up one frame per iteration budget and then captures physics when the
substep count is even. The graphs share the simulation states.
Use ``--disable-cuda-graph`` to compare ordinary stepping; CPU and odd-substep runs use ordinary
stepping automatically. The recorded validation included USD recording and concurrent GPU load;
it is a correctness qualification, rather than an idle-GPU real-time performance benchmark.

The default robot asset is pinned to Newton Assets revision
``a96f0973b6ae69c90609f9fafef9d2c1db2d6431`` so that hand dimensions, inertias, and TCP geometry
remain reproducible independently of Newton updates.

Softer stocking validation
--------------------------

The second iteration reduced bending stiffness by 40 times and bending damping by 100 times,
retains the original stretch stiffness, and raises the supported opening by 100 mm. Full-surface
contacts were already enabled in the initial prototype; the additional closed wrapper collider
makes those contacts include the decorative tails. Newton uses exact mesh feature contacts for
this path, without requiring a volume SDF.

The six-second, three-candy free-drop check passed on the same Newton revision and RTX 5090:
all three cores and sampled wrapper surfaces remained contained, cuff drift was zero, maximum
final cloth-edge stretch was 1.059, and minimum final bench clearance was 101 mm. Relative to
the empty drape at matching times, the largest cloth-vertex displacement was 35.7 mm; the final
RMS difference was 7.2 mm. These measurements demonstrate a load response, not a calibration
against a particular real stocking. Eight focused geometry, motion, and export tests pass.

The full 44.7-second, three-candy filling sequence also passed with those second-iteration settings and
exact mesh contacts, without an SDF. All cores and sampled wrapper surfaces remained contained;
peak TCP error was 5.66 mm, cuff drift was zero, maximum final cloth-edge stretch was 1.033,
and final bench clearance was 106 mm. A 30-frame CPU mesh-contact smoke run completed with
finite states; it is not a full CPU sequence qualification.

Garment rest shape and damping
------------------------------

The third iteration flattened the rest cross-section below the supported cuff with each ring's
fabric circumference preserved. It lowered bending stiffness 100 times and raised bending
damping 10 times relative to the second iteration.
The material-only comparison on the old geometry reduced last-second RMS cloth speed in a
six-second empty drape from 15.0 to 3.0 mm/s. Damping reduces transient motion; it does not
replace a suitable garment rest shape.

Useful upstream references are Newton's `VBD tuning guide
<https://newton-physics.github.io/newton/latest/concepts/simulation_tuning_vbd.html>`_ and its
`cloth bags with rigid contents
<https://github.com/newton-physics/newton/blob/main/newton/examples/vbd/example_vbd_soft_rigid_contact.py>`_.
The bag example is a contact reference, not a wool material calibration. Newton's H1 garment
example uses Style3D, whose material coefficients should not be copied directly into VBD.
Lightwheel's public leisaac cloth-folding scene uses a particle-system asset; a ready-made
VBD wool sweater preset was not found in the public sources examined.

The third-iteration mesh and material passed the original six-second three-candy drop check. All
cores and sampled wrapper surfaces were contained at the end, cuff drift was zero, maximum
final edge stretch was 1.240, and final bench clearance was 85.5 mm. The full Franka sequence
has not been rerun for this third iteration. Its visual comparison and measurement records
are separate from the second iteration's full-sequence qualification.

That third-iteration check measured final geometry but did not require settling. Inspection of
the recording found sustained cloth speeds around 0.24–0.30 m/s after impact despite containment.
It is not accepted as stable motion. The current check additionally requires a complete,
finite final-second velocity window below 0.03 m/s RMS; this samples physics velocity each
rendered frame and does not qualify substep-scale vibration.

Settling qualification
---------------------

The flattened, very soft mesh needs more solve iterations than the original rounded mesh.
With the geometry and contacts fixed, 60 iterations reduced the recorded final-second speed
to about 31 mm/s. Raising bending damping alone to 0.001 left about 109 mm/s, while raising
membrane damping alone to 1.0 left 94 mm/s. Neither damping-only change passed the settling
criterion. The empty flattened mesh also failed it, excluding candy contact as the sole cause.

The fourth iteration combined 60 iterations with bending damping 0.001, preserving bending
stiffness 0.0000005 and membrane damping 0.01. The complete six-second, three-candy drop test
passed the new settling criterion with 7.53 mm/s final-second RMS cloth speed. All cores and
sampled wrapper surfaces were contained. The higher iteration count increases solver work
per frame.

Selected stocking
-----------------

Option B from the shape and damping grid uses the existing flattened mesh with bending damping
0.004 and 60 VBD iterations. B retained all three candies in the six-second drop test, with
1.62 mm/s final-second RMS cloth speed and 92.6 mm of final bench clearance.

Using 60 iterations throughout the first arm run caused all three candy cores to slip from
the gripper during lift. The script caps the solve at 15 iterations during pickup and carry.
During finger opening, targets of at least 18 mm per finger select the cloth iteration budget
for impact and settling. One VBD solver advances all bodies and cloth throughout the sequence.

The selected configuration passed the full 44.7-second Franka sequence on an RTX 5090 with
the pinned Newton revision. All three cores and sampled wrapper surfaces stayed contained at
the end. Peak TCP tracking error was 4.18 mm, maximum final edge stretch was 1.196, bench
clearance was 89.8 mm, and final-second RMS cloth speed was 1.80 mm/s.

Sewn stocking preview
---------------------

Select the two-panel pattern with ``--stocking-pattern sewn``. Both panels share vertices
along the sides and curved sole. The shaped heel and instep define the foot in the flat
pattern. Below the open cuff the unloaded panels are about 14 mm apart at mid-leg.
This version uses the same hoop, material settings, candies and collision settings as B.
The shape was informed by the finished stocking and construction photographs in
`Studio Knit's chunky stocking tutorial <https://www.studioknitsf.com/chunky-christmas-stocking/>`_.

.. code-block:: bash

   atelier --stocking-pattern sewn --mode drop --candies 3 --device cuda:0 --viewer usd --check

Faster sewn-stocking drops
--------------------------

For rapid free-drop previews, two substeps per 60 Hz frame and 60 iterations per substep
passed two six-second, three-candy runs on the full 1,728-triangle garment. All three cores
and sampled wrappers were inside at the end of each run. Final-second RMS cloth speeds were
23.4 and 8.0 mm/s. One substep with 60 iterations and two substeps with 30 or 45 iterations
produced more than twofold edge stretch within the first second.

.. code-block:: bash

   atelier --stocking-pattern sewn --mode drop --substeps 2 --iterations 60 --device cuda:0 --viewer null --check

The selected sewn-stocking drop defaults to 12 substeps and 15 iterations. In the
recorded run the final cloth vertices differed from the 12-by-60 reference by 5.8 mm RMS.
The measured foot depth was 43 mm, compared with 49 mm in the reference and 74 mm in the
recorded two-substep run. All three configurations use the same mesh, material and face contacts.

Full sewn-stocking fill
-----------------------

An earlier run with 12 substeps and 15 iterations lost the second candy during transfer.
The current VBD coloring puts the candies in a dedicated solve group after the robot's groups,
so the fingers and candies update in separate passes. With that change, the 44.7 s CUDA run
passed at 12 substeps and 15 iterations using the full 1,728-triangle sewn stocking and capsule
cores. All three cores and sampled wrapper surfaces stayed inside. Peak TCP error was 2.42 mm,
and final-second RMS cloth speed was 6.10 mm/s.

``repro_franka_grasp.py`` provides a separate flat-ground grasp comparison for capsules,
cylinders, and cubes using VBD or MuJoCo. Run it with ``--help`` for the available contact and
solver settings.
