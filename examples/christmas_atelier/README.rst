Christmas Atelier
=================

Watch a scripted Franka FR3 pick three wrapped candies from a tray and drop them into
a sewn stocking. Run from an Isaac Lab checkout after the standard uv setup.

.. code-block:: bash

   uv run isaaclab demo christmas-atelier

In an installed Isaac Lab environment, run ``isaaclab demo christmas-atelier``.
The first launch downloads the Franka assets. The demo opens the Newton GL visualizer.
Pass ``--viz none`` to run headless.

Isaac Lab runs the robot, candies, and cloth in one Newton VBD solver. The stocking and
wrappers are generated in ``assets.py``. The scripted fill takes 44.7 seconds of simulated time.
