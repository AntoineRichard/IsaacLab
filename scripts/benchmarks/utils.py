# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal benchmark utilities used by benchmark entry-point scripts."""


def get_backend_type(cli_backend: str) -> str:
    """Map old CLI backend names to new backend types.

    Args:
        cli_backend: The backend name from CLI arguments.

    Returns:
        The new backend type string.
    """
    mapping = {
        "OmniPerfKPIFile": "omniperf",
        "JSONFileMetrics": "json",
        "OsmoKPIFile": "osmo",
        "LocalLogMetrics": "json",
        "omniperf": "omniperf",
        "json": "json",
        "osmo": "osmo",
        "summary": "summary",
    }
    return mapping.get(cli_backend, "omniperf")
