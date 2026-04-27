# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Odin CUDA detection + per-host upgrade helpers.

Public surface mirrors ``tools/odin/asgard/bootstrap.py``: a single-host
function and a fleet driver per phase (``check`` / ``install``).
"""

from __future__ import annotations

import re

__all__ = [
    "cuda_at_or_above",
    "parse_nvidia_smi",
    "parse_os_release",
]


_NVIDIA_SMI_HEADER_RE = re.compile(
    r"NVIDIA-SMI\s+\S+\s+Driver Version:\s+(?P<driver>\d+\.\d+(?:\.\d+)?)\s+CUDA Version:\s+(?P<cuda>\d+\.\d+)"
)


def parse_nvidia_smi(stdout: str) -> tuple[str, str] | None:
    """Extract ``(driver_version, cuda_version)`` from ``nvidia-smi`` output.

    The header line on every supported driver release looks like::

        | NVIDIA-SMI 535.161.07             Driver Version: 535.161.07   CUDA Version: 12.2     |

    We pull the driver version (``"535.161.07"``) and the driver-advertised
    max CUDA (``"12.2"``).

    Args:
        stdout: Raw stdout from ``nvidia-smi`` (or any string).

    Returns:
        Two-tuple of ``(driver, cuda)`` strings, or ``None`` if no header
        line is found (e.g. ``nvidia-smi`` is missing or ran on a no-GPU host).
    """
    m = _NVIDIA_SMI_HEADER_RE.search(stdout)
    if m is None:
        return None
    return (m.group("driver"), m.group("cuda"))


_OS_RELEASE_RE = re.compile(r'(?m)^(?P<key>[A-Z_]+)=(?:"(?P<qval>[^"]*)"|(?P<val>\S*))')


def parse_os_release(stdout: str) -> str | None:
    """Map the contents of ``/etc/os-release`` to NVIDIA's apt repo slug.

    Returns ``"ubuntu2204"`` or ``"ubuntu2404"`` on supported Ubuntus,
    ``None`` for everything else (RHEL, Debian, no file).

    Args:
        stdout: Raw contents of ``/etc/os-release``.
    """
    fields: dict[str, str] = {}
    for m in _OS_RELEASE_RE.finditer(stdout):
        fields[m.group("key")] = m.group("qval") if m.group("qval") is not None else m.group("val") or ""
    if fields.get("ID") != "ubuntu":
        return None
    version_id = fields.get("VERSION_ID", "")
    if version_id == "22.04":
        return "ubuntu2204"
    if version_id == "24.04":
        return "ubuntu2404"
    return None


def cuda_at_or_above(measured: str, floor: str) -> bool:
    """Return ``True`` iff the dotted CUDA string ``measured >= floor``.

    Numeric (not lexicographic) comparison: ``"12.10" >= "12.4"`` is True.
    Both inputs must be ``"<major>.<minor>"`` — ``ValueError`` otherwise.

    Args:
        measured: Observed CUDA version string (e.g. ``"12.2"``).
        floor: Minimum required CUDA version string (e.g. ``"12.4"``).
    """
    return _parse_cuda(measured) >= _parse_cuda(floor)


def _parse_cuda(s: str) -> tuple[int, int]:
    parts = s.split(".")
    if len(parts) != 2:
        raise ValueError(f"expected 'major.minor' CUDA string, got {s!r}")
    try:
        return (int(parts[0]), int(parts[1]))
    except ValueError as e:
        raise ValueError(f"expected 'major.minor' CUDA string, got {s!r}") from e
