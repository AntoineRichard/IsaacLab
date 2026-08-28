# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Convert the MicroDuck walk MJCF into the single USD file shipped with ``isaaclab_assets``.

This wraps the same :class:`~isaaclab.sim.converters.MjcfConverter` that ``convert_mjcf.py``
drives, and only adds what that script's flags cannot express:

* it selects the ``"physx"`` entry of the generated ``"Physics"`` variant set, which is the only
  variant Isaac Lab's Newton importer reads the MJCF joint armature from (Newton resolves
  ``physxJoint:armature``; the ``mjc:*`` attributes of the ``"mujoco"`` variant are not in the
  resolver set Isaac Lab passes to ``ModelBuilder.add_usd``, and that variant also drops the
  drive force range);
* it flattens the layered asset the importer emits (an interface layer plus seven payloads) into
  one self-contained binary USD, so the shipped asset is a single file.

Usage:

.. code-block:: bash

    uv run python scripts/tools/convert_microduck.py <path/to/robot_walk.xml>

See ``ATTRIBUTION.md`` next to the generated asset for the provenance of the source MJCF.
"""

"""Parse CLI first so we can decide whether to launch Isaac Sim Kit."""

import argparse
import os

from isaaclab.app import AppLauncher, add_launcher_args, launch_simulation
from isaaclab.utils.version import standalone_importers_available

_DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "source",
    "isaaclab_assets",
    "data",
    "Robots",
    "PollenRobotics",
    "MicroDuck",
    "microduck_walk.usd",
)

parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
parser.add_argument("input", type=str, help="The path to the MicroDuck ``robot_walk.xml`` MJCF file.")
parser.add_argument(
    "--output",
    type=str,
    default=_DEFAULT_OUTPUT,
    help="The path to store the flattened USD file. Defaults to the asset shipped with isaaclab_assets.",
)
add_launcher_args(parser)
args_cli = parser.parse_args()

# Prefer kit-less: the standalone importer wheel runs the same importer without starting Kit.
args_cli.require_kit = not standalone_importers_available()
args_cli.physics = "isaacsim_physx" if args_cli.require_kit else "newton_mjwarp"

if args_cli.require_kit and not AppLauncher.is_available():
    raise ImportError(
        "MJCF conversion requires either the full Isaac Sim runtime or the standalone"
        " 'isaacsim-asset-isolated' importer wheel, but neither is installed."
    )

import tempfile  # noqa: E402

from pxr import Usd  # noqa: E402

from isaaclab.physics import PhysicsCfg  # noqa: E402
from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg  # noqa: E402
from isaaclab.utils.assets import check_file_path  # noqa: E402


def flatten_to_single_file(layered_usd_path: str, dest_path: str) -> None:
    """Compose the layered asset the MJCF importer emits into one binary USD file.

    The importer writes an interface layer that payloads geometry, materials and physics from
    sibling files. Flattening bakes the composed result — including the selected ``"Physics"``
    variant — into a single layer, and keeps the mesh prototypes so the instanced visual geometry
    is not duplicated.

    Args:
        layered_usd_path: Path of the interface layer written by the importer.
        dest_path: Path of the single USD file to write.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    Usd.Stage.Open(layered_usd_path).Flatten().Export(dest_path)


def main():
    mjcf_path = os.path.abspath(args_cli.input)
    if not check_file_path(mjcf_path):
        raise ValueError(f"Invalid file path: {mjcf_path}")
    dest_path = os.path.abspath(args_cli.output)

    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli):
        # The layered asset is an intermediate: only the flattened file is shipped, and keeping the
        # scratch copy out of the source tree stops a stale one from shadowing a re-conversion.
        with tempfile.TemporaryDirectory(prefix="microduck_mjcf_") as scratch_dir:
            converter = MjcfConverter(
                MjcfConverterCfg(
                    asset_path=mjcf_path,
                    usd_dir=scratch_dir,
                    force_usd_conversion=True,
                    physics_variant=MjcfConverterCfg.PhysicsVariant.PHYSX,
                )
            )
            flatten_to_single_file(converter.usd_path, dest_path)

    print(f"Converted {mjcf_path} to {dest_path}")


if __name__ == "__main__":
    main()
