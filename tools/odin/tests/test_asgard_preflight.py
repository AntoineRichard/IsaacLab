# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :func:`tools.odin.asgard.preflight.preflight_valkyrie`."""

from __future__ import annotations

from dataclasses import dataclass

from tools.odin.asgard.fleet import ValkyrieConfig
from tools.odin.asgard.preflight import PreflightResult, preflight_valkyrie
from tools.odin.asgard.transport import SSHResult


@dataclass
class _FakeSSH:
    """Deterministic SSH runner: returns scripted SSHResult for each cmd substring match."""

    scripted: dict  # {cmd_substring: SSHResult}

    def run(self, host, cmd: str, *, timeout_s=None, stdout_tee=None, pty=True) -> SSHResult:
        for key, result in self.scripted.items():
            if key in cmd:
                return result
        return SSHResult(exit_code=1, stdout="", stderr=f"no fake for {cmd!r}", duration_s=0.0)


def _host() -> ValkyrieConfig:
    return ValkyrieConfig(host="v1", ssh_user="odin")


def _ok() -> SSHResult:
    return SSHResult(exit_code=0, stdout="ok\n", stderr="", duration_s=0.01)


def _fail(msg: str) -> SSHResult:
    return SSHResult(exit_code=1, stdout="", stderr=msg, duration_s=0.01)


def test_all_checks_pass():
    ssh = _FakeSSH(
        scripted={
            "echo preflight-ok": _ok(),
            "docker ps": _ok(),
            "docker inspect": SSHResult(exit_code=0, stdout="running\n", stderr="", duration_s=0.01),
            "test -d": _ok(),
            "nvidia-smi -L": SSHResult(
                exit_code=0,
                stdout="GPU 0: NVIDIA A100\n",
                stderr="",
                duration_s=0.01,
            ),
        }
    )
    r = preflight_valkyrie(_host(), ssh=ssh)
    assert isinstance(r, PreflightResult)
    assert r.ok is True
    assert r.checks == {
        "ssh_reach": True,
        "docker_running": True,
        "container_up": True,
        "isaaclab_present": True,
        "gpu_present": True,
    }


def test_ssh_unreachable():
    ssh = _FakeSSH(scripted={"echo preflight-ok": _fail("connection refused")})
    r = preflight_valkyrie(_host(), ssh=ssh)
    assert r.ok is False
    assert r.checks["ssh_reach"] is False
    # Downstream checks should NOT be run / should be False.
    assert r.checks["docker_running"] is False
    assert "connection refused" in r.message


def test_docker_down():
    ssh = _FakeSSH(
        scripted={
            "echo preflight-ok": _ok(),
            "docker ps": _fail("docker daemon not responding"),
        }
    )
    r = preflight_valkyrie(_host(), ssh=ssh)
    assert r.ok is False
    assert r.checks["ssh_reach"] is True
    assert r.checks["docker_running"] is False
    assert "docker" in r.message.lower()


def test_container_down():
    ssh = _FakeSSH(
        scripted={
            "echo preflight-ok": _ok(),
            "docker ps": _ok(),
            "docker inspect": SSHResult(exit_code=0, stdout="exited\n", stderr="", duration_s=0.01),
            "test -d": _ok(),
        }
    )
    r = preflight_valkyrie(_host(), ssh=ssh)
    assert r.ok is False
    assert r.checks["container_up"] is False
    assert "container" in r.message.lower()


def test_isaaclab_missing():
    ssh = _FakeSSH(
        scripted={
            "echo preflight-ok": _ok(),
            "docker ps": _ok(),
            "docker inspect": SSHResult(exit_code=0, stdout="running\n", stderr="", duration_s=0.01),
            "test -d": _fail("no such directory"),
        }
    )
    r = preflight_valkyrie(_host(), ssh=ssh)
    assert r.ok is False
    assert r.checks["isaaclab_present"] is False


def test_all_checks_pass_with_gpu():
    """Five-check happy path: ssh + docker + container + isaaclab + gpu."""
    ssh = _FakeSSH(
        scripted={
            "echo preflight-ok": _ok(),
            "docker ps": _ok(),
            "docker inspect": SSHResult(exit_code=0, stdout="running\n", stderr="", duration_s=0.01),
            "test -d": _ok(),
            "nvidia-smi -L": SSHResult(
                exit_code=0,
                stdout="GPU 0: NVIDIA A100 (UUID: GPU-abc...)\n",
                stderr="",
                duration_s=0.01,
            ),
        }
    )
    r = preflight_valkyrie(_host(), ssh=ssh)
    assert r.ok is True
    assert r.checks == {
        "ssh_reach": True,
        "docker_running": True,
        "container_up": True,
        "isaaclab_present": True,
        "gpu_present": True,
    }


def test_healthy_host_has_no_recovery_flags():
    """A host that passes all checks on first probe has both recovery flags False."""
    ssh = _FakeSSH(
        scripted={
            "echo preflight-ok": _ok(),
            "docker ps": _ok(),
            "docker inspect": SSHResult(exit_code=0, stdout="running\n", stderr="", duration_s=0.01),
            "test -d": _ok(),
            "nvidia-smi -L": SSHResult(
                exit_code=0,
                stdout="GPU 0: NVIDIA A100\n",
                stderr="",
                duration_s=0.01,
            ),
        }
    )
    r = preflight_valkyrie(_host(), ssh=ssh)
    assert r.ok is True
    assert r.recovery_attempted is False
    assert r.recovery_succeeded is False


def test_gpu_absent_marks_host_down():
    """nvidia-smi -L returns non-zero → preflight fails with gpu_present=False."""
    ssh = _FakeSSH(
        scripted={
            "echo preflight-ok": _ok(),
            "docker ps": _ok(),
            "docker inspect": SSHResult(exit_code=0, stdout="running\n", stderr="", duration_s=0.01),
            "test -d": _ok(),
            "nvidia-smi -L": SSHResult(
                exit_code=255,
                stdout="",
                stderr="Failed to initialize NVML: Unknown Error",
                duration_s=0.01,
            ),
        }
    )
    r = preflight_valkyrie(_host(), ssh=ssh, auto_restart=False)
    assert r.ok is False
    assert r.checks["gpu_present"] is False
    assert "gpu" in r.message.lower()


def test_gpu_present_short_circuits_on_container_down():
    """If container_up fails, gpu_present stays False without probe call."""
    call_count = {"n": 0}

    class _CountingSSH:
        def run(self, host, cmd, *, timeout_s=None, stdout_tee=None, pty=True):
            call_count["n"] += 1
            if "echo preflight-ok" in cmd:
                return _ok()
            if "docker ps" in cmd:
                return _ok()
            if "docker inspect" in cmd:
                return SSHResult(exit_code=0, stdout="exited\n", stderr="", duration_s=0.01)
            if "nvidia-smi -L" in cmd:
                # Should never reach here.
                raise AssertionError("nvidia-smi -L called despite container down")
            return _ok()

    r = preflight_valkyrie(_host(), ssh=_CountingSSH())
    assert r.ok is False
    assert r.checks["gpu_present"] is False
    assert r.checks["container_up"] is False


def test_preflight_recovers_nvml_wedge_via_container_restart():
    """First nvidia-smi shows NVML init failure → preflight calls
    recover_valkyrie_gpu → second nvidia-smi succeeds → host marked healthy."""
    nvml_fail = SSHResult(
        exit_code=255,
        stdout="",
        stderr="Failed to initialize NVML: Unknown Error",
        duration_s=0.01,
    )
    nvml_ok = SSHResult(exit_code=0, stdout="GPU 0: NVIDIA L40\n", stderr="", duration_s=0.01)

    class _SequencedSSH:
        def __init__(self, scripted, nvidia_seq):
            self.scripted = scripted
            self.nvidia_seq = list(nvidia_seq)

        def run(self, host, cmd, *, timeout_s=None, stdout_tee=None, pty=True):
            if "nvidia-smi -L" in cmd:
                return self.nvidia_seq.pop(0)
            for k, v in self.scripted.items():
                if k in cmd:
                    return v
            return SSHResult(exit_code=1, stdout="", stderr="", duration_s=0.0)

    ssh = _SequencedSSH(
        scripted={
            "echo preflight-ok": _ok(),
            "docker ps": _ok(),
            "docker inspect": SSHResult(exit_code=0, stdout="running\n", stderr="", duration_s=0.01),
            "test -d": _ok(),
            "docker restart": _ok(),
        },
        # Three nvidia-smi -L probes happen on the recovery happy path:
        # (1) preflight's initial check, (2) recover_valkyrie_gpu's internal
        # acceptance probe, (3) preflight's post-recover re-probe for the
        # ``parse_gpu_class`` lookup.
        nvidia_seq=[nvml_fail, nvml_ok, nvml_ok],
    )
    r = preflight_valkyrie(_host(), ssh=ssh, auto_restart=True)
    assert r.ok is True
    assert r.checks["gpu_present"] is True
    assert r.message == "recovered: container restarted to clear NVML wedge"
    assert r.recovery_attempted is True
    assert r.recovery_succeeded is True


def test_preflight_no_auto_restart_marks_host_down():
    """auto_restart=False preserves the strict-failure semantic."""
    ssh = _FakeSSH(
        scripted={
            "echo preflight-ok": _ok(),
            "docker ps": _ok(),
            "docker inspect": SSHResult(exit_code=0, stdout="running\n", stderr="", duration_s=0.01),
            "test -d": _ok(),
            "nvidia-smi -L": SSHResult(
                exit_code=255,
                stdout="",
                stderr="Failed to initialize NVML: Unknown Error",
                duration_s=0.01,
            ),
        }
    )
    r = preflight_valkyrie(_host(), ssh=ssh, auto_restart=False)
    assert r.ok is False
    assert r.checks["gpu_present"] is False
    assert r.recovery_attempted is False


def test_preflight_recovers_on_ssh_disconnect_during_nvidia_smi():
    """SSH connection drops mid-``nvidia-smi -L`` (non-zero exit, stderr like
    ``"Connection to <host> closed."``, possibly non-empty stdout) → preflight
    must still attempt one container-restart recovery.

    Regression for the 2026-05-05 incident where two hosts wedged with that
    exact signature, the prior stderr-signature gate did not match, and
    SSH-banner bytes on stdout defeated the ``not r.stdout.strip()`` clause —
    so ``recovery_attempted`` stayed False and the dispatcher crashed at the
    2/5-host preflight wall instead of self-healing.

    Contract: by the time preflight reaches the GPU check, all four upstream
    checks (ssh_reach, docker, container, isaaclab path) have already passed,
    so any nvidia-smi failure is a sign that the GPU view in the container
    has gone south. ``auto_restart=True`` should always try one container
    restart; the specific failure signature should not gate the attempt."""
    ssh_disconnect = SSHResult(
        exit_code=255,
        # SSH-banner / pty noise on stdout (e.g., MOTD remnants, control bytes)
        # — non-empty so the prior ``not r.stdout.strip()`` clause was False.
        stdout="kex_exchange_identification: read: Connection reset by peer\n",
        stderr="Connection to v1 closed.\n",
        duration_s=0.01,
    )
    nvml_ok = SSHResult(exit_code=0, stdout="GPU 0: NVIDIA L40\n", stderr="", duration_s=0.01)

    class _SequencedSSH:
        def __init__(self, scripted, nvidia_seq):
            self.scripted = scripted
            self.nvidia_seq = list(nvidia_seq)

        def run(self, host, cmd, *, timeout_s=None, stdout_tee=None, pty=True):
            if "nvidia-smi -L" in cmd:
                return self.nvidia_seq.pop(0)
            for k, v in self.scripted.items():
                if k in cmd:
                    return v
            return SSHResult(exit_code=1, stdout="", stderr="", duration_s=0.0)

    ssh = _SequencedSSH(
        scripted={
            "echo preflight-ok": _ok(),
            "docker ps": _ok(),
            "docker inspect": SSHResult(exit_code=0, stdout="running\n", stderr="", duration_s=0.01),
            "test -d": _ok(),
            "docker restart": _ok(),
        },
        # Three nvidia-smi probes (initial, recovery's internal acceptance,
        # post-recover gpu_class re-probe) — see the wedge test above.
        nvidia_seq=[ssh_disconnect, nvml_ok, nvml_ok],
    )
    r = preflight_valkyrie(_host(), ssh=ssh, auto_restart=True)
    assert r.ok is True
    assert r.checks["gpu_present"] is True
    assert r.recovery_attempted is True
    assert r.recovery_succeeded is True


def test_preflight_recovery_failure_marks_host_down():
    """auto_restart=True but second nvidia-smi still fails → host down with
    a recovery_failed marker."""
    nvml_fail = SSHResult(
        exit_code=255,
        stdout="",
        stderr="Failed to initialize NVML: Unknown Error",
        duration_s=0.01,
    )

    class _SequencedSSH:
        def __init__(self, scripted, nvidia_seq):
            self.scripted = scripted
            self.nvidia_seq = list(nvidia_seq)

        def run(self, host, cmd, *, timeout_s=None, stdout_tee=None, pty=True):
            if "nvidia-smi -L" in cmd:
                return self.nvidia_seq.pop(0)
            for k, v in self.scripted.items():
                if k in cmd:
                    return v
            return SSHResult(exit_code=1, stdout="", stderr="", duration_s=0.0)

    ssh = _SequencedSSH(
        scripted={
            "echo preflight-ok": _ok(),
            "docker ps": _ok(),
            "docker inspect": SSHResult(exit_code=0, stdout="running\n", stderr="", duration_s=0.01),
            "test -d": _ok(),
            "docker restart": _ok(),
        },
        nvidia_seq=[nvml_fail, nvml_fail],
    )
    r = preflight_valkyrie(_host(), ssh=ssh, auto_restart=True)
    assert r.ok is False
    assert "recovery_failed" in r.message
    assert r.recovery_attempted is True
    assert r.recovery_succeeded is False


def test_preflight_records_cuda_version():
    """Healthy host's CUDA version is captured from driver query."""
    ssh = _FakeSSH(
        scripted={
            "echo preflight-ok": _ok(),
            "docker ps": _ok(),
            "docker inspect": SSHResult(exit_code=0, stdout="running\n", stderr="", duration_s=0.01),
            "test -d": _ok(),
            "nvidia-smi -L": SSHResult(exit_code=0, stdout="GPU 0: L40\n", stderr="", duration_s=0.01),
            "nvidia-smi --query-gpu=driver_version": SSHResult(
                exit_code=0,
                stdout="535.161.07\n",
                stderr="",
                duration_s=0.01,
            ),
        }
    )
    r = preflight_valkyrie(_host(), ssh=ssh)
    assert r.cuda_version is not None
    assert r.cuda_version == (12, 2)
    # No floor passed → newton always available.
    assert r.newton_available is True


def test_preflight_marks_newton_unavailable_below_floor():
    """Driver 535 → CUDA 12.2; floor 12.4 → newton_available=False, but ok=True (physx still works)."""
    ssh = _FakeSSH(
        scripted={
            "echo preflight-ok": _ok(),
            "docker ps": _ok(),
            "docker inspect": SSHResult(exit_code=0, stdout="running\n", stderr="", duration_s=0.01),
            "test -d": _ok(),
            "nvidia-smi -L": SSHResult(exit_code=0, stdout="GPU 0: L40\n", stderr="", duration_s=0.01),
            "nvidia-smi --query-gpu=driver_version": SSHResult(
                exit_code=0,
                stdout="535.161.07\n",
                stderr="",
                duration_s=0.01,
            ),
        }
    )
    r = preflight_valkyrie(_host(), ssh=ssh, newton_cuda_floor=(12, 4))
    assert r.ok is True  # available for physx
    assert r.cuda_version == (12, 2)
    assert r.newton_available is False
    assert "newton floor 12.4" in r.message
    assert "odin-cuda install --target 12.4" in r.message


def test_preflight_above_floor_keeps_newton_available():
    """Driver 570 → CUDA 12.8 ≥ floor 12.4 → newton_available=True."""
    ssh = _FakeSSH(
        scripted={
            "echo preflight-ok": _ok(),
            "docker ps": _ok(),
            "docker inspect": SSHResult(exit_code=0, stdout="running\n", stderr="", duration_s=0.01),
            "test -d": _ok(),
            "nvidia-smi -L": SSHResult(exit_code=0, stdout="GPU 0: H100\n", stderr="", duration_s=0.01),
            "nvidia-smi --query-gpu=driver_version": SSHResult(
                exit_code=0,
                stdout="570.0.0\n",
                stderr="",
                duration_s=0.01,
            ),
        }
    )
    r = preflight_valkyrie(_host(), ssh=ssh, newton_cuda_floor=(12, 4))
    assert r.ok is True
    assert r.cuda_version == (12, 8)
    assert r.newton_available is True
    assert r.message == ""
