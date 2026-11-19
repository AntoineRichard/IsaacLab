Christmas stocking filling atelier
==================================

A scripted Franka FR3 picks three wrapped candies from a tray and drops them into a sewn
Christmas stocking. Run the demo from an Isaac Lab checkout after the standard uv setup.

.. code-block:: bash

   uv run isaaclab demo christmas-atelier

The demo opens Isaac Lab's Newton GL visualizer. Pass ``--viz none`` for a headless run.
The first launch downloads the pinned FR3 and Franka hand asset bundle. Use
``--robot-urdf PATH`` with a local copy of the same URDF and its meshes for offline runs.

Recording and checking a run
----------------------------

.. code-block:: bash

   uv run isaaclab demo christmas-atelier --viz none --device cuda:0 \
       --output /tmp/christmas-atelier.usd --metrics /tmp/christmas-atelier.json --check

``--output`` records animated USD. ``--metrics`` records grasp and release measurements,
cloth settling, and final containment. ``--check`` returns a nonzero status if the run ends
early, a pick or release fails, a candy or wrapper escapes, or the stocking fails its
stability checks. The full sequence takes 44.7 seconds of simulated time.

The tests cover mesh boundaries, asset export, motion and containment, and the headless
drape path.

.. code-block:: bash

   uv run python -m pytest examples/christmas_atelier/tests -q

Scene and control
-----------------

Isaac Lab's ``SimulationContext`` owns the Newton backend, collision pipeline, and visualizer.
The workcell is built procedurally into the backend's shared builder. One VBD solver advances
the FR3, the wrapped candies, and the cloth. The 20 g candy core collides with the fingers
and tray. A closed wrapper collider catches cloth faces and edges, including contact with
the twisted tails. The red and cream panels share vertices at the seams. The brass rim
supports the open cuff while the rest of the cloth deforms freely.

The demo plans end-effector waypoints once on a separate IK model and writes interpolated
joint targets to Isaac Lab's live Newton backend. Both fingers have physical drives. VBD
solves the candy bodies in a separate group after the fingers so grasp contacts use updated
finger poses. The fingertips release the candy above the opening and gravity carries it into
the stocking.

Tuning
------

Use ``--mode drape`` for an empty stocking, ``--mode drop`` for a free drop, or
``--mode probe`` for a single scripted grasp and release without the cloth.
``--max_steps`` limits any mode to a short run. The default sewn stocking uses a
28-by-32 mesh, 12 physics substeps per 60 Hz frame, and 15 VBD iterations per substep.
Use ``--help`` to see the mesh, fabric, and drive settings.

.. code-block:: bash

   uv run isaaclab demo christmas-atelier --mode drop --viz none --check
   uv run isaaclab demo christmas-atelier --export-assets /tmp/christmas-assets

The asset export writes the custom stocking and wrapper geometry in meters. The animated
USD and interactive viewer use the scene's base colors. The showcase render adds knit
material and lighting in Blender.
