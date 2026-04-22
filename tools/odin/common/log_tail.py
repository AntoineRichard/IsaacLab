# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Log tail utility: capture the last N bytes of a stream to a file."""

from __future__ import annotations

# Default tail size per the T1 spec: 16 KB per stream.
DEFAULT_TAIL_BYTES = 16 * 1024


def tail_bytes(data: bytes, max_bytes: int = DEFAULT_TAIL_BYTES) -> bytes:
    """Return the last ``max_bytes`` of ``data``, or all of it if shorter."""
    if len(data) <= max_bytes:
        return data
    return data[-max_bytes:]
