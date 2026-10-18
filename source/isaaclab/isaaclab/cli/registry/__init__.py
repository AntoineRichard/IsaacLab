# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Client for the Isaac Lab registry of community task packages (prototype).

The registry is a curated index of template-generated Isaac Lab projects that lives in a separate,
bot-run repository. Clients only download its compiled ``index.json``. This package reads that index,
gates releases on the running Isaac Lab release, and installs them with uv, either into the user's own
uv project (``uv add --group isaaclab-registry``) or into the running environment (``uv pip install``).

Every module here imports only the standard library at module level, so ``import isaaclab.cli`` keeps
working before Isaac Lab's runtime dependencies are installed.
"""

import re
import sys

from ..utils import print_warning

DEPENDENCY_GROUP = "isaaclab-registry"
"""Dependency group that holds registry packages in a user's uv project."""

TASKS_ENTRY_POINT_GROUP = "isaaclab.tasks"
"""Entry-point group through which installed projects register their tasks."""

SECURITY_WARNING = (
    "Registry packages run third-party code at install time and every time Isaac Lab loads tasks. Review the source."
)
"""Warning printed before every install."""


class RegistryError(RuntimeError):
    """A registry operation failed with a message meant for the user."""


def warn(message: str) -> None:
    """Print a warning on stderr, apart from the tables on stdout."""
    print_warning(message, stream=sys.stderr)


def normalize_name(name: str) -> str:
    """Return the PEP 503 normalized form of a distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def is_isaaclab_distribution(name: str) -> bool:
    """Return whether a distribution is Isaac Lab itself or one of its ``isaaclab-*`` packages."""
    normalized = normalize_name(name)
    return normalized == "isaaclab" or normalized.startswith("isaaclab-")
