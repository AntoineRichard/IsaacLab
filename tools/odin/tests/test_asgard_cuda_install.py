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


# --- check_cuda_valkyrie ---------------------------------------------------

from dataclasses import dataclass, field

from tools.odin.asgard.cuda_install import CheckResult, check_cuda_valkyrie
from tools.odin.asgard.fleet import ValkyrieConfig


@dataclass
class _SSHCall:
    cmd: str
    timeout_s: float | None


@dataclass
class _FakeSSH:
    """Records calls; replies via substring lookup on ``cmd`` (insertion-ordered)."""

    calls: list[_SSHCall] = field(default_factory=list)
    replies: dict[str, int] = field(default_factory=dict)
    reply_stdout: dict[str, str] = field(default_factory=dict)
    reply_stderr: dict[str, str] = field(default_factory=dict)
    reply_timed_out: dict[str, bool] = field(default_factory=dict)

    def run(self, host, cmd, *, timeout_s=None, stdout_tee=None):
        self.calls.append(_SSHCall(cmd=cmd, timeout_s=timeout_s))
        exit_code = 0
        stdout = ""
        stderr = ""
        timed_out = False
        for key, code in self.replies.items():
            if key in cmd:
                exit_code = code
                stdout = self.reply_stdout.get(key, "")
                stderr = self.reply_stderr.get(key, "")
                timed_out = self.reply_timed_out.get(key, False)
                break

        class R:
            pass

        R.exit_code = exit_code
        R.stdout = stdout
        R.stderr = stderr
        R.duration_s = 0.01
        R.timed_out = timed_out
        return R()


def _host() -> ValkyrieConfig:
    return ValkyrieConfig(
        host="v1.internal",
        ssh_user="odin",
        ssh_key=None,
        isaaclab_path="/opt/IsaacLab",
        container_name="isaac-lab-base",
    )


_NVIDIA_SMI_OK_122 = "| NVIDIA-SMI 535.161.07             Driver Version: 535.161.07   CUDA Version: 12.2     |"
_NVIDIA_SMI_OK_129 = "| NVIDIA-SMI 575.51.03             Driver Version: 575.51.03   CUDA Version: 12.9     |"
_OS_2404 = 'ID=ubuntu\nVERSION_ID="24.04"\n'


def test_check_ok_when_at_or_above_floor():
    ssh = _FakeSSH(
        replies={
            "echo cuda-check-ok": 0,
            "nvidia-smi": 0,
            "/etc/os-release": 0,
        },
        reply_stdout={"nvidia-smi": _NVIDIA_SMI_OK_129, "/etc/os-release": _OS_2404},
    )
    result = check_cuda_valkyrie(_host(), ssh=ssh, floor="12.4")
    assert isinstance(result, CheckResult)
    assert result.host == "v1.internal"
    assert result.status == "ok"
    assert result.driver == "575.51.03"
    assert result.cuda == "12.9"


def test_check_needs_upgrade_when_below_floor():
    ssh = _FakeSSH(
        replies={"nvidia-smi": 0, "/etc/os-release": 0},
        reply_stdout={"nvidia-smi": _NVIDIA_SMI_OK_122, "/etc/os-release": _OS_2404},
    )
    result = check_cuda_valkyrie(_host(), ssh=ssh, floor="12.4")
    assert result.status == "needs-upgrade"
    assert result.cuda == "12.2"


def test_check_unreachable_when_ssh_probe_fails():
    ssh = _FakeSSH(
        replies={"echo cuda-check-ok": 255},
        reply_stderr={"echo cuda-check-ok": "ssh: connect to host: No route"},
    )
    result = check_cuda_valkyrie(_host(), ssh=ssh, floor="12.4")
    assert result.status == "unreachable"
    assert "No route" in result.message
    # nvidia-smi must NOT have been attempted.
    assert not any("nvidia-smi" in c.cmd for c in ssh.calls)


def test_check_no_gpu_when_nvidia_smi_missing():
    ssh = _FakeSSH(
        replies={"nvidia-smi": 127},
        reply_stderr={"nvidia-smi": "nvidia-smi: command not found"},
    )
    result = check_cuda_valkyrie(_host(), ssh=ssh, floor="12.4")
    assert result.status == "no-gpu"


def test_check_no_gpu_when_nvidia_smi_unparseable():
    ssh = _FakeSSH(
        replies={"nvidia-smi": 0},
        reply_stdout={"nvidia-smi": "(garbled output, no header line)"},
    )
    result = check_cuda_valkyrie(_host(), ssh=ssh, floor="12.4")
    assert result.status == "no-gpu"


def test_check_unsupported_os_short_circuits():
    ssh = _FakeSSH(
        replies={"nvidia-smi": 0, "/etc/os-release": 0},
        reply_stdout={
            "nvidia-smi": _NVIDIA_SMI_OK_122,
            "/etc/os-release": 'ID=rhel\nVERSION_ID="9.4"\n',
        },
    )
    result = check_cuda_valkyrie(_host(), ssh=ssh, floor="12.4")
    assert result.status == "unsupported-os"
    # We still report observed driver/cuda for context.
    assert result.driver == "535.161.07"
    assert result.cuda == "12.2"
