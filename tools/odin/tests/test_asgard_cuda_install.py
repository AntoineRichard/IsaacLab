# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for tools.odin.asgard.cuda_install."""

from __future__ import annotations

import pytest

from tools.odin.asgard.cuda_install import (
    cuda_at_or_above,
    parse_nvidia_smi,
    parse_os_release,
)


# --- parse_nvidia_smi -------------------------------------------------------


_SMI_OK = """\
Mon Apr 27 11:52:21 2026
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.161.07             Driver Version: 535.161.07   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
"""


def test_parse_nvidia_smi_extracts_driver_and_cuda():
    parsed = parse_nvidia_smi(_SMI_OK)
    assert parsed == ("535.161.07", "12.2")


def test_parse_nvidia_smi_returns_none_when_no_match():
    assert parse_nvidia_smi("nvidia-smi: command not found") is None
    assert parse_nvidia_smi("") is None


def test_parse_nvidia_smi_handles_long_driver_minor():
    line = "| NVIDIA-SMI 575.51.03             Driver Version: 575.51.03   CUDA Version: 12.9     |"
    assert parse_nvidia_smi(line) == ("575.51.03", "12.9")


# --- parse_os_release -------------------------------------------------------


_OS_2404 = """\
PRETTY_NAME="Ubuntu 24.04.3 LTS"
NAME="Ubuntu"
VERSION_ID="24.04"
ID=ubuntu
VERSION_CODENAME=noble
"""

_OS_2204 = """\
NAME="Ubuntu"
VERSION_ID="22.04"
ID=ubuntu
"""

_OS_RHEL = """\
NAME="Red Hat Enterprise Linux"
VERSION_ID="9.4"
ID=rhel
"""


def test_parse_os_release_ubuntu_2404():
    assert parse_os_release(_OS_2404) == "ubuntu2404"


def test_parse_os_release_ubuntu_2204():
    assert parse_os_release(_OS_2204) == "ubuntu2204"


def test_parse_os_release_unsupported_returns_none():
    assert parse_os_release(_OS_RHEL) is None
    assert parse_os_release("") is None


# --- cuda_at_or_above -------------------------------------------------------


@pytest.mark.parametrize(
    "measured,floor,expected",
    [
        ("12.2", "12.4", False),
        ("12.4", "12.4", True),
        ("12.9", "12.4", True),
        ("13.0", "12.4", True),
        ("11.8", "12.4", False),
        ("12.10", "12.4", True),  # numeric, not lexicographic
    ],
)
def test_cuda_at_or_above(measured: str, floor: str, expected: bool):
    assert cuda_at_or_above(measured, floor) is expected


def test_cuda_at_or_above_rejects_garbage():
    with pytest.raises(ValueError):
        cuda_at_or_above("12", "12.4")
    with pytest.raises(ValueError):
        cuda_at_or_above("not.a.version", "12.4")
