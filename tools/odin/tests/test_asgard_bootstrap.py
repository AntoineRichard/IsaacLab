# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for valhalla-adjacent Asgard bootstrap (bring fresh Valkyries up)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from tools.odin.asgard.bootstrap import BootstrapResult, bootstrap_valkyrie
from tools.odin.asgard.fleet import ValkyrieConfig

# --- Fakes -----------------------------------------------------------------


@dataclass
class _SSHCall:
    cmd: str
    timeout_s: float | None


@dataclass
class _RsyncCall:
    local_path: Path
    remote_path: str


@dataclass
class _FakeSSH:
    """Records calls; replies with a per-call exit_code lookup.

    Default reply is exit_code=0. Override by setting ``replies[key]`` where
    ``key`` is a substring that must appear in the cmd. First match wins;
    check order follows insertion order.
    """

    calls: list[_SSHCall] = field(default_factory=list)
    replies: dict[str, int] = field(default_factory=dict)
    reply_stdout: dict[str, str] = field(default_factory=dict)
    reply_stderr: dict[str, str] = field(default_factory=dict)

    def run(self, host, cmd, *, timeout_s=None, stdout_tee=None):
        self.calls.append(_SSHCall(cmd=cmd, timeout_s=timeout_s))
        exit_code = 0
        stdout = ""
        stderr = ""
        for key, code in self.replies.items():
            if key in cmd:
                exit_code = code
                stdout = self.reply_stdout.get(key, "")
                stderr = self.reply_stderr.get(key, "")
                break

        class R:
            pass

        R.exit_code = exit_code
        R.stdout = stdout
        R.stderr = stderr
        R.duration_s = 0.01
        return R()


@dataclass
class _FakeRsync:
    calls: list[_RsyncCall] = field(default_factory=list)
    exit_code: int = 0
    stderr: str = ""

    def push(self, host, local_path, remote_path):
        self.calls.append(_RsyncCall(local_path=Path(local_path), remote_path=str(remote_path)))

        class R:
            pass

        R.exit_code = self.exit_code
        R.stdout = ""
        R.stderr = self.stderr
        R.duration_s = 0.01
        return R()

    def pull(self, host, remote_path, local_path):
        raise AssertionError("bootstrap_valkyrie must not call rsync.pull")


def _host() -> ValkyrieConfig:
    return ValkyrieConfig(
        host="v1.internal",
        ssh_user="odin",
        ssh_key=None,
        isaaclab_path="/opt/IsaacLab",
        container_name="isaac-lab-base",
    )


# --- Tests -----------------------------------------------------------------


def test_bootstrap_valkyrie_happy_path(tmp_path: Path):
    ssh = _FakeSSH(
        replies={"docker inspect": 0},
        reply_stdout={"docker inspect": "running"},
    )
    rsync = _FakeRsync()
    result = bootstrap_valkyrie(_host(), tmp_path, ssh=ssh, rsync=rsync)
    assert isinstance(result, BootstrapResult)
    assert result.ok is True
    assert result.host == "v1.internal"
    assert set(result.step_durations_s.keys()) == {
        "wipe",
        "rsync",
        "configure_headless",
        "create_odin_runs",
        "container_start",
        "fix_isaac_sim_symlink",
        "container_verify",
    }
    assert all(d >= 0.0 for d in result.step_durations_s.values())


def test_bootstrap_valkyrie_ssh_unreachable(tmp_path: Path):
    ssh = _FakeSSH(replies={"echo bootstrap-ok": 255})
    rsync = _FakeRsync()
    result = bootstrap_valkyrie(_host(), tmp_path, ssh=ssh, rsync=rsync)
    assert result.ok is False
    assert "ssh unreachable" in result.message
    assert rsync.calls == [], "rsync.push must not run when ssh is down"


def test_bootstrap_valkyrie_docker_daemon_down(tmp_path: Path):
    ssh = _FakeSSH(
        replies={"docker ps": 1},
        reply_stderr={"docker ps": "Cannot connect to the Docker daemon"},
    )
    rsync = _FakeRsync()
    result = bootstrap_valkyrie(_host(), tmp_path, ssh=ssh, rsync=rsync)
    assert result.ok is False
    assert "docker daemon" in result.message
    assert rsync.calls == []


def test_bootstrap_valkyrie_rsync_failure(tmp_path: Path):
    ssh = _FakeSSH()  # all ssh ok
    rsync = _FakeRsync(exit_code=23, stderr="send_files failed")
    result = bootstrap_valkyrie(_host(), tmp_path, ssh=ssh, rsync=rsync)
    assert result.ok is False
    assert "rsync push failed" in result.message
    # Wipe ran; rsync ran; container_start did NOT.
    assert any("rm -rf" in c.cmd for c in ssh.calls)
    assert len(rsync.calls) == 1
    assert not any("container.py start" in c.cmd for c in ssh.calls)
    # configure_headless must NOT run either when rsync fails.
    assert not any(".container.cfg" in c.cmd for c in ssh.calls)


def test_bootstrap_valkyrie_writes_headless_cfg(tmp_path: Path):
    """configure_headless step writes X11_FORWARDING_ENABLED=0 to remote .container.cfg."""
    ssh = _FakeSSH(
        replies={"docker inspect": 0},
        reply_stdout={"docker inspect": "running"},
    )
    rsync = _FakeRsync()
    bootstrap_valkyrie(_host(), tmp_path, ssh=ssh, rsync=rsync)
    cfg_calls = [c for c in ssh.calls if ".container.cfg" in c.cmd]
    assert len(cfg_calls) == 1, f"expected exactly one .container.cfg write, got {len(cfg_calls)}"
    assert "X11_FORWARDING_ENABLED = 0" in cfg_calls[0].cmd
    assert "docker/.container.cfg" in cfg_calls[0].cmd


def test_bootstrap_valkyrie_configure_headless_failure(tmp_path: Path):
    """When the .container.cfg write fails, bootstrap stops before container.py start."""
    ssh = _FakeSSH(
        replies={".container.cfg": 1},
        reply_stderr={".container.cfg": "permission denied"},
    )
    rsync = _FakeRsync()
    result = bootstrap_valkyrie(_host(), tmp_path, ssh=ssh, rsync=rsync)
    assert result.ok is False
    assert "headless .container.cfg" in result.message
    assert not any("container.py start" in c.cmd for c in ssh.calls)


def test_bootstrap_valkyrie_container_start_failure(tmp_path: Path):
    ssh = _FakeSSH(
        replies={"container.py start": 1},
        reply_stderr={"container.py start": "timeout"},
    )
    rsync = _FakeRsync()
    result = bootstrap_valkyrie(_host(), tmp_path, ssh=ssh, rsync=rsync)
    assert result.ok is False
    assert "container.py start" in result.message or "container start failed" in result.message
    assert not any("docker inspect" in c.cmd for c in ssh.calls), "verify must not run when start failed"


def test_bootstrap_valkyrie_container_not_running_after_start(tmp_path: Path):
    ssh = _FakeSSH(
        replies={"docker inspect": 0},
        reply_stdout={"docker inspect": "exited"},
    )
    rsync = _FakeRsync()
    result = bootstrap_valkyrie(_host(), tmp_path, ssh=ssh, rsync=rsync)
    assert result.ok is False
    assert "not running" in result.message
    assert "'exited'" in result.message


def test_bootstrap_valkyrie_build_timeout_passed_through(tmp_path: Path):
    ssh = _FakeSSH(
        replies={"docker inspect": 0},
        reply_stdout={"docker inspect": "running"},
    )
    rsync = _FakeRsync()
    bootstrap_valkyrie(_host(), tmp_path, ssh=ssh, rsync=rsync, build_timeout_s=1800)
    start_calls = [c for c in ssh.calls if "container.py start" in c.cmd]
    assert len(start_calls) == 1
    assert start_calls[0].timeout_s == 1800


def test_bootstrap_valkyrie_wipe_always_runs(tmp_path: Path):
    ssh = _FakeSSH(
        replies={"docker inspect": 0},
        reply_stdout={"docker inspect": "running"},
    )
    rsync = _FakeRsync()
    bootstrap_valkyrie(_host(), tmp_path, ssh=ssh, rsync=rsync)
    wipe_calls = [c for c in ssh.calls if "rm -rf" in c.cmd and "/opt/IsaacLab" in c.cmd]
    assert len(wipe_calls) == 1


def test_bootstrap_valkyrie_wipe_failure(tmp_path: Path):
    """Wipe-step failure (e.g. permission issue) halts the pipeline before rsync."""
    ssh = _FakeSSH(
        replies={"rm -rf": 1},
        reply_stderr={"rm -rf": "Permission denied"},
    )
    rsync = _FakeRsync()
    result = bootstrap_valkyrie(_host(), tmp_path, ssh=ssh, rsync=rsync)
    assert result.ok is False
    assert "failed to wipe" in result.message
    assert "/opt/IsaacLab" in result.message
    assert rsync.calls == [], "rsync.push must not run when wipe failed"


def test_bootstrap_valkyrie_container_inspect_fails_surfaces_stderr(tmp_path: Path):
    """When docker inspect returns non-zero (e.g. no such container), stderr must surface."""
    ssh = _FakeSSH(
        replies={"docker inspect": 1},
        reply_stderr={"docker inspect": "Error: No such object: isaac-lab-base"},
    )
    rsync = _FakeRsync()
    result = bootstrap_valkyrie(_host(), tmp_path, ssh=ssh, rsync=rsync)
    assert result.ok is False
    assert "docker inspect failed" in result.message
    assert "No such object" in result.message


def test_bootstrap_fleet_returns_results_in_fleet_order(tmp_path: Path):
    from tools.odin.asgard.bootstrap import bootstrap_fleet
    from tools.odin.asgard.fleet import Fleet

    fleet = Fleet(
        fleet_name="test",
        hosts=[
            ValkyrieConfig(host="v1", ssh_user="u", ssh_key=None, isaaclab_path="/p"),
            ValkyrieConfig(host="v2", ssh_user="u", ssh_key=None, isaaclab_path="/p"),
            ValkyrieConfig(host="v3", ssh_user="u", ssh_key=None, isaaclab_path="/p"),
        ],
    )
    ssh = _FakeSSH(replies={"docker inspect": 0}, reply_stdout={"docker inspect": "running"})
    rsync = _FakeRsync()
    results = bootstrap_fleet(fleet, tmp_path, ssh=ssh, rsync=rsync, parallel=False)
    assert [r.host for r in results] == ["v1", "v2", "v3"]
    assert all(r.ok for r in results)


def test_bootstrap_fleet_mixed_outcome(tmp_path: Path):
    """One host reaches SSH fine, another fails; both appear in the result list."""
    from tools.odin.asgard.bootstrap import bootstrap_fleet
    from tools.odin.asgard.fleet import Fleet

    fleet = Fleet(
        fleet_name="test",
        hosts=[
            ValkyrieConfig(host="v-good", ssh_user="u", ssh_key=None, isaaclab_path="/p"),
            ValkyrieConfig(host="v-bad", ssh_user="u", ssh_key=None, isaaclab_path="/p"),
        ],
    )

    # A per-host SSH fake: v-bad's first probe fails; v-good otherwise normal.
    good_ssh = _FakeSSH(replies={"docker inspect": 0}, reply_stdout={"docker inspect": "running"})
    bad_ssh = _FakeSSH(replies={"echo bootstrap-ok": 255}, reply_stderr={"echo bootstrap-ok": "conn refused"})

    # Wrap both with a routing SSH that dispatches on host.host.
    @dataclass
    class _RoutingSSH:
        def run(self, host, cmd, *, timeout_s=None, stdout_tee=None):
            inner = good_ssh if host.host == "v-good" else bad_ssh
            return inner.run(host, cmd, timeout_s=timeout_s, stdout_tee=stdout_tee)

    results = bootstrap_fleet(fleet, tmp_path, ssh=_RoutingSSH(), rsync=_FakeRsync(), parallel=False)
    assert len(results) == 2
    good = next(r for r in results if r.host == "v-good")
    bad = next(r for r in results if r.host == "v-bad")
    assert good.ok is True
    assert bad.ok is False
    assert "ssh unreachable" in bad.message


def test_bootstrap_fleet_parallel_runs_concurrently(tmp_path: Path):
    """With 3 hosts and parallel=True, wall time ≈ max(per-host), not sum."""
    import time as _time_mod

    from tools.odin.asgard.bootstrap import bootstrap_fleet
    from tools.odin.asgard.fleet import Fleet

    fleet = Fleet(
        fleet_name="test",
        hosts=[ValkyrieConfig(host=f"v{i}", ssh_user="u", ssh_key=None, isaaclab_path="/p") for i in (1, 2, 3)],
    )

    # SSH fake that sleeps 100 ms on container.py start to simulate slow hosts.
    class _SlowSSH(_FakeSSH):
        def run(self, host, cmd, *, timeout_s=None, stdout_tee=None):
            if "container.py start" in cmd:
                _time_mod.sleep(0.1)
            return super().run(host, cmd, timeout_s=timeout_s, stdout_tee=stdout_tee)

    ssh = _SlowSSH(replies={"docker inspect": 0}, reply_stdout={"docker inspect": "running"})
    rsync = _FakeRsync()

    t0 = _time_mod.perf_counter()
    results = bootstrap_fleet(fleet, tmp_path, ssh=ssh, rsync=rsync, parallel=True)
    elapsed = _time_mod.perf_counter() - t0

    assert all(r.ok for r in results)
    # Serial would be ≥ 3 * 0.1 = 0.3 s. Parallel should be < 0.25 s.
    assert elapsed < 0.25, f"parallel=True elapsed={elapsed:.3f}s (expected <0.25)"


def test_bootstrap_fleet_sequential_adds_up(tmp_path: Path):
    """With parallel=False, wall time grows linearly with host count."""
    import time as _time_mod

    from tools.odin.asgard.bootstrap import bootstrap_fleet
    from tools.odin.asgard.fleet import Fleet

    fleet = Fleet(
        fleet_name="test",
        hosts=[ValkyrieConfig(host=f"v{i}", ssh_user="u", ssh_key=None, isaaclab_path="/p") for i in (1, 2, 3)],
    )

    class _SlowSSH(_FakeSSH):
        def run(self, host, cmd, *, timeout_s=None, stdout_tee=None):
            if "container.py start" in cmd:
                _time_mod.sleep(0.1)
            return super().run(host, cmd, timeout_s=timeout_s, stdout_tee=stdout_tee)

    ssh = _SlowSSH(replies={"docker inspect": 0}, reply_stdout={"docker inspect": "running"})
    rsync = _FakeRsync()

    t0 = _time_mod.perf_counter()
    bootstrap_fleet(fleet, tmp_path, ssh=ssh, rsync=rsync, parallel=False)
    elapsed = _time_mod.perf_counter() - t0

    # Serial wall time ≥ 3 * 0.1 s — allow loose upper bound for scheduler noise.
    assert elapsed >= 0.28, f"parallel=False elapsed={elapsed:.3f}s (expected >=0.28)"


def test_bootstrap_fleet_verbose_prints_per_host(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    from tools.odin.asgard.bootstrap import bootstrap_fleet
    from tools.odin.asgard.fleet import Fleet

    fleet = Fleet(
        fleet_name="test",
        hosts=[ValkyrieConfig(host="v-only", ssh_user="u", ssh_key=None, isaaclab_path="/p")],
    )
    ssh = _FakeSSH(replies={"docker inspect": 0}, reply_stdout={"docker inspect": "running"})
    rsync = _FakeRsync()
    bootstrap_fleet(fleet, tmp_path, ssh=ssh, rsync=rsync, parallel=False, verbose=True)
    out = capsys.readouterr().out
    assert "v-only" in out
    assert "ok" in out


def test_bootstrap_valkyrie_creates_odin_runs_dir(tmp_path: Path):
    """bootstrap must `mkdir -p {isaaclab_path}/odin_runs` after configure_headless."""
    ssh = _FakeSSH(
        replies={"docker inspect": 0},
        reply_stdout={"docker inspect": "running"},
    )
    rsync = _FakeRsync()
    bootstrap_valkyrie(_host(), tmp_path, ssh=ssh, rsync=rsync)
    mkdir_calls = [c for c in ssh.calls if "mkdir -p" in c.cmd and "/opt/IsaacLab/odin_runs" in c.cmd]
    assert len(mkdir_calls) == 1, f"expected exactly one mkdir for odin_runs, got {len(mkdir_calls)}"


def test_bootstrap_valkyrie_create_odin_runs_failure(tmp_path: Path):
    """A failed `mkdir -p odin_runs` stops the pipeline before container.py start."""
    ssh = _FakeSSH(
        replies={"mkdir -p /opt/IsaacLab/odin_runs": 1},
        reply_stderr={"mkdir -p /opt/IsaacLab/odin_runs": "Permission denied"},
    )
    rsync = _FakeRsync()
    result = bootstrap_valkyrie(_host(), tmp_path, ssh=ssh, rsync=rsync)
    assert result.ok is False
    assert "create" in result.message and "odin_runs" in result.message
    assert not any("container.py start" in c.cmd for c in ssh.calls)
