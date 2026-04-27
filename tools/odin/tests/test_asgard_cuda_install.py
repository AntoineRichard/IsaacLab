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


# --- check_fleet ----------------------------------------------------------

from tools.odin.asgard.cuda_install import check_fleet
from tools.odin.asgard.fleet import Fleet


def test_check_fleet_returns_results_in_fleet_order():
    fleet = Fleet(
        fleet_name="test",
        hosts=[
            ValkyrieConfig(host="v1", ssh_user="u", ssh_key=None, isaaclab_path="/p"),
            ValkyrieConfig(host="v2", ssh_user="u", ssh_key=None, isaaclab_path="/p"),
            ValkyrieConfig(host="v3", ssh_user="u", ssh_key=None, isaaclab_path="/p"),
        ],
    )
    ssh = _FakeSSH(
        replies={"nvidia-smi": 0, "/etc/os-release": 0},
        reply_stdout={"nvidia-smi": _NVIDIA_SMI_OK_129, "/etc/os-release": _OS_2404},
    )
    results = check_fleet(fleet, ssh=ssh, floor="12.4", parallel=False)
    assert [r.host for r in results] == ["v1", "v2", "v3"]
    assert all(r.status == "ok" for r in results)


def test_check_fleet_parallel_runs_concurrently():
    import time as _t

    fleet = Fleet(
        fleet_name="test",
        hosts=[ValkyrieConfig(host=f"v{i}", ssh_user="u", ssh_key=None, isaaclab_path="/p") for i in (1, 2, 3)],
    )

    class _SlowSSH(_FakeSSH):
        def run(self, host, cmd, *, timeout_s=None, stdout_tee=None):
            if "nvidia-smi" in cmd:
                _t.sleep(0.1)
            return super().run(host, cmd, timeout_s=timeout_s, stdout_tee=stdout_tee)

    ssh = _SlowSSH(
        replies={"nvidia-smi": 0, "/etc/os-release": 0},
        reply_stdout={"nvidia-smi": _NVIDIA_SMI_OK_129, "/etc/os-release": _OS_2404},
    )
    t0 = _t.perf_counter()
    results = check_fleet(fleet, ssh=ssh, floor="12.4", parallel=True)
    elapsed = _t.perf_counter() - t0
    assert all(r.status == "ok" for r in results)
    assert elapsed < 0.25, f"parallel check_fleet elapsed={elapsed:.3f}s"


# --- install_cuda_valkyrie skip path --------------------------------------

from tools.odin.asgard.cuda_install import (
    TARGET_TO_DRIVER_MAJOR,
    CudaInstallResult,
    install_cuda_valkyrie,
)


def test_target_to_driver_major_has_pinned_default():
    # The default install target must be present.
    assert TARGET_TO_DRIVER_MAJOR["12.9"] == "575"


def test_install_skip_when_already_at_floor():
    ssh = _FakeSSH(
        replies={"nvidia-smi": 0, "/etc/os-release": 0},
        reply_stdout={"nvidia-smi": _NVIDIA_SMI_OK_129, "/etc/os-release": _OS_2404},
    )
    result = install_cuda_valkyrie(_host(), ssh=ssh, floor="12.4", target="12.9")
    assert isinstance(result, CudaInstallResult)
    assert result.ok is True
    assert result.skipped is True
    assert result.driver_before == "575.51.03"
    assert result.cuda_before == "12.9"
    # Skip path must not run apt or reboot.
    assert not any("apt-get install" in c.cmd for c in ssh.calls)
    assert not any("systemctl reboot" in c.cmd for c in ssh.calls)


def test_install_skip_unreachable_propagates_as_failure():
    ssh = _FakeSSH(
        replies={"echo cuda-check-ok": 255},
        reply_stderr={"echo cuda-check-ok": "ssh: conn refused"},
    )
    result = install_cuda_valkyrie(_host(), ssh=ssh, floor="12.4", target="12.9")
    assert result.ok is False
    assert result.skipped is False
    assert "ssh unreachable" in result.message


def test_install_unsupported_os_refuses():
    ssh = _FakeSSH(
        replies={"nvidia-smi": 0, "/etc/os-release": 0},
        reply_stdout={"nvidia-smi": _NVIDIA_SMI_OK_122, "/etc/os-release": 'ID=rhel\nVERSION_ID="9.4"\n'},
    )
    result = install_cuda_valkyrie(_host(), ssh=ssh, floor="12.4", target="12.9")
    assert result.ok is False
    assert "Ubuntu" in result.message
    # No install attempt.
    assert not any("apt-get" in c.cmd for c in ssh.calls)


def test_install_unknown_target_raises():
    ssh = _FakeSSH()
    with pytest.raises(ValueError, match="unknown target"):
        install_cuda_valkyrie(_host(), ssh=ssh, floor="12.4", target="99.99")


# --- install_cuda_valkyrie full pipeline ----------------------------------


def _stub_clock_and_sleep():
    """Return ``(clock, sleep)`` injectables that advance a fake monotonic
    counter without ever calling time.sleep — tests stay sub-second."""
    now = [0.0]

    def _clock() -> float:
        return now[0]

    def _sleep(delta: float) -> None:
        now[0] += delta

    return _clock, _sleep


def _install_happy_path_ssh() -> _FakeSSH:
    """SSH fake where pre-check sees 12.2, post-verify sees 12.9."""

    @dataclass
    class _PrePostSSH(_FakeSSH):
        post_phase: bool = False

        def run(self, host, cmd, *, timeout_s=None, stdout_tee=None):
            r = super().run(host, cmd, timeout_s=timeout_s, stdout_tee=stdout_tee)
            # Flip to "post" the moment the install asks for the reboot.
            if "systemctl reboot" in cmd:
                self.post_phase = True
            return r

        # Override stdout for nvidia-smi based on phase.
        def __post_init__(self):
            self.replies.setdefault("echo cuda-check-ok", 0)
            self.replies.setdefault("/etc/os-release", 0)
            self.replies.setdefault("nvidia-smi", 0)
            self.replies.setdefault("container.py stop", 0)
            self.replies.setdefault("cuda-keyring", 0)
            self.replies.setdefault("apt-get update", 0)
            self.replies.setdefault("apt-get install", 0)
            self.replies.setdefault("systemctl reboot", 0)
            self.replies.setdefault("echo cuda-install-ok", 0)
            self.replies.setdefault("container.py start", 0)
            self.reply_stdout.setdefault("/etc/os-release", _OS_2404)

    ssh = _PrePostSSH()
    # Phase-dependent nvidia-smi stdout.
    base_run = ssh.run

    def _phased_run(host, cmd, *, timeout_s=None, stdout_tee=None):
        if "nvidia-smi" in cmd:
            ssh.reply_stdout["nvidia-smi"] = _NVIDIA_SMI_OK_129 if ssh.post_phase else _NVIDIA_SMI_OK_122
        return base_run(host, cmd, timeout_s=timeout_s, stdout_tee=stdout_tee)

    ssh.run = _phased_run  # type: ignore[assignment]
    return ssh


def test_install_full_happy_path_runs_all_steps():
    ssh = _install_happy_path_ssh()
    clock, sleep = _stub_clock_and_sleep()
    result = install_cuda_valkyrie(
        _host(),
        ssh=ssh,
        floor="12.4",
        target="12.9",
        reboot_timeout_s=600.0,
        clock=clock,
        sleep=sleep,
    )
    assert result.ok is True
    assert result.skipped is False
    assert result.cuda_before == "12.2"
    assert result.cuda_after == "12.9"
    assert result.driver_after.startswith("575.")
    # Step set must include the install pipeline phases.
    assert {
        "container_stop",
        "add_repo",
        "apt_update",
        "apt_install",
        "reboot",
        "wait_for_ssh",
        "post_verify",
        "container_start",
    } <= set(result.step_durations_s)


def test_install_apt_install_failure_short_circuits():
    ssh = _install_happy_path_ssh()
    ssh.replies["apt-get install"] = 100
    ssh.reply_stderr["apt-get install"] = "E: Unable to locate package cuda-12-9"
    clock, sleep = _stub_clock_and_sleep()
    result = install_cuda_valkyrie(_host(), ssh=ssh, floor="12.4", target="12.9", clock=clock, sleep=sleep)
    assert result.ok is False
    assert "apt" in result.message.lower()
    assert not any("systemctl reboot" in c.cmd for c in ssh.calls)


def test_install_reboot_timeout_is_hard_failure():
    ssh = _install_happy_path_ssh()
    # Make every "echo cuda-install-ok" probe non-zero to simulate the host
    # never coming back.
    ssh.replies["echo cuda-install-ok"] = 255
    clock, sleep = _stub_clock_and_sleep()
    result = install_cuda_valkyrie(
        _host(),
        ssh=ssh,
        floor="12.4",
        target="12.9",
        reboot_timeout_s=30.0,
        clock=clock,
        sleep=sleep,
    )
    assert result.ok is False
    assert "reboot timed out" in result.message
    assert not any("container.py start" in c.cmd for c in ssh.calls if "stop" not in c.cmd)


def test_install_post_verify_failure_when_cuda_still_below_floor():
    """Reboot succeeds but driver kmod didn't load; nvidia-smi still reports 12.2."""
    ssh = _FakeSSH(
        replies={
            "echo cuda-check-ok": 0,
            "/etc/os-release": 0,
            "nvidia-smi": 0,
            "container.py stop": 0,
            "cuda-keyring": 0,
            "apt-get update": 0,
            "apt-get install": 0,
            "systemctl reboot": 0,
            "echo cuda-install-ok": 0,
            "container.py start": 0,
        },
        reply_stdout={
            "nvidia-smi": _NVIDIA_SMI_OK_122,  # never flips
            "/etc/os-release": _OS_2404,
        },
    )
    clock, sleep = _stub_clock_and_sleep()
    result = install_cuda_valkyrie(_host(), ssh=ssh, floor="12.4", target="12.9", clock=clock, sleep=sleep)
    assert result.ok is False
    assert "verify" in result.message.lower()


def test_install_container_stop_failure_is_non_fatal():
    """A failed container.py stop must not block the install."""
    ssh = _install_happy_path_ssh()
    ssh.replies["container.py stop"] = 1
    clock, sleep = _stub_clock_and_sleep()
    result = install_cuda_valkyrie(_host(), ssh=ssh, floor="12.4", target="12.9", clock=clock, sleep=sleep)
    assert result.ok is True


def test_install_container_start_failure_is_soft_warning():
    """A failed container.py start after install yields ok=True with a message."""
    ssh = _install_happy_path_ssh()
    ssh.replies["container.py start"] = 1
    ssh.reply_stderr["container.py start"] = "image build error"
    clock, sleep = _stub_clock_and_sleep()
    result = install_cuda_valkyrie(_host(), ssh=ssh, floor="12.4", target="12.9", clock=clock, sleep=sleep)
    assert result.ok is True
    assert "container restart" in result.message
