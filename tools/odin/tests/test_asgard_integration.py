# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Slow-marked integration test: Asgard dispatch against ssh localhost.

Replaces the 'docker exec' command with a shell stub so the test runs
without docker or Isaac Sim. Still exercises the real ShellSSHRunner and
ShellRsyncRunner subprocess paths, covering the transport + dispatch wiring
end-to-end.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from tools.odin.asgard.fleet import Fleet, ValkyrieConfig
from tools.odin.asgard.runner import DispatchOptions, run_dispatch
from tools.odin.asgard.transport import ShellRsyncRunner, ShellSSHRunner
from tools.odin.common.env_list import EnvEntry, EnvList, write_env_list


def _ssh_localhost_works() -> bool:
    """Probe: can we `ssh localhost "echo ok"` without a password?"""
    try:
        r = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "localhost", "echo ok"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return r.returncode == 0 and "ok" in r.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


pytestmark = pytest.mark.slow


@pytest.fixture
def stub_ssh_runner(monkeypatch, tmp_path: Path):
    """Replace ShellSSHRunner.run's docker-exec command with a local stub that
    materialises a valid bundle and exits 0."""
    from tools.odin.asgard import worker as worker_mod

    real_build = worker_mod._build_docker_exec_cmd

    def _fake_build(host: ValkyrieConfig, job) -> str:
        # Write a minimal valid bundle into the host's odin_runs/ directory
        # (same path the real runner + rsync pull target expects).
        bundle_dir = f"{host.isaaclab_path}/odin_runs/{job.bundle_dir_name}"
        manifest = {
            "schema_version": "1.0",
            "phases": {"training": {"status": "completed"}, "startup": {"status": "completed"}},
        }
        training = {"schema_version": "1.0"}
        startup = {"schema_version": "1.0"}
        manifest_s = json.dumps(manifest).replace("'", r"\'")
        training_s = json.dumps(training).replace("'", r"\'")
        startup_s = json.dumps(startup).replace("'", r"\'")
        return (
            f"mkdir -p {bundle_dir} && "
            f"printf '%s' '{manifest_s}' > {bundle_dir}/manifest.json && "
            f"printf '%s' '{training_s}' > {bundle_dir}/training.json && "
            f"printf '%s' '{startup_s}' > {bundle_dir}/startup.json"
        )

    monkeypatch.setattr(worker_mod, "_build_docker_exec_cmd", _fake_build)
    yield
    monkeypatch.setattr(worker_mod, "_build_docker_exec_cmd", real_build)


@pytest.fixture
def stub_provisioner(monkeypatch):
    """Preflight (docker ps / docker inspect) would fail on vanilla localhost.
    Short-circuit preflight and provisioner to always pass."""
    from tools.odin.asgard import preflight as pf
    from tools.odin.asgard import provisioner as pv

    def _fake_pf(host, *, ssh, auto_restart=True, newton_cuda_floor=None):
        return pf.PreflightResult(
            host=host.host,
            ok=True,
            checks={"ssh_reach": True, "docker_running": True, "container_up": True, "isaaclab_present": True},
            message="",
        )

    def _fake_pv(host, working_tree, *, fresh, ssh, rsync):
        return pv.ProvisionResult(host=host.host, ok=True, commit_sha="integration-stub")

    monkeypatch.setattr("tools.odin.asgard.runner.preflight_valkyrie", _fake_pf)
    monkeypatch.setattr("tools.odin.asgard.runner.provision_valkyrie", _fake_pv)


def test_loopback_dispatch_against_localhost(tmp_path: Path, stub_ssh_runner, stub_provisioner):
    if not _ssh_localhost_works():
        pytest.skip("ssh localhost does not work without a password; skipping integration test")

    # Build a one-row env list.
    el = EnvList()
    el.groups["direct/ant"] = [
        EnvEntry(
            task_id="Isaac-Ant-Direct-v0",
            entry_point="isaaclab_tasks.direct.ant:AntEnv",
            env_cfg_entry_point="isaaclab_tasks.direct.ant.ant_env_cfg:AntEnvCfg",
            group="direct/ant",
            has_rsl_rl=True,
            has_skrl=True,
            has_rl_games=False,
            framework="rsl_rl",
            num_envs=4096,
            max_iterations=10,
            keep=True,
            status="current",
        )
    ]
    physx_yaml = tmp_path / "physx.yaml"
    write_env_list(physx_yaml, el, generator="test")

    # One-host fleet targeting localhost; use the current repo as the remote
    # "isaaclab_path" so the fake bundle lands in a path we can rsync back.
    repo_root = Path.cwd()
    host = ValkyrieConfig(
        host="localhost",
        ssh_user=os.environ.get("USER", "root"),
        isaaclab_path=str(repo_root),
    )
    fleet = Fleet(fleet_name="loopback-test", hosts=[host])
    dispatch_dir = tmp_path / "odin_runs" / "20260422-loopback"
    dispatch_dir.mkdir(parents=True)

    state = run_dispatch(
        fleet=fleet,
        physx_yaml=physx_yaml,
        newton_yaml=None,
        dispatch_dir=dispatch_dir,
        options=DispatchOptions(seeds=[42], per_job_timeout_s=60, detached_mode=False),
        ssh=ShellSSHRunner(),
        rsync=ShellRsyncRunner(),
    )

    # The one job must complete.
    assert len(state.jobs) == 1
    assert state.jobs[0].status == "completed", f"job failed: {state.jobs[0].failure}"

    # Bundle must have been pulled back to the dispatch directory.
    bundle = dispatch_dir / state.jobs[0].bundle_dir_name
    assert (bundle / "manifest.json").exists()
    assert (bundle / "logs" / "ssh-tail.log").exists()

    # dispatch.json must record success.
    dj = json.loads((dispatch_dir / "dispatch.json").read_text())
    assert dj["jobs"][0]["status"] == "completed"

    # Spec 0 / Task 7: aggregator now also writes hardware.json.
    # The loopback _build_docker_exec stub doesn't populate
    # training.json.hardware on every kernel, so accept absence —
    # but if present, validate the schema shape.
    hw_path = dispatch_dir / "hardware.json"
    if hw_path.exists():
        hw = json.loads(hw_path.read_text())
        assert hw["schema_version"] == "1.0"
        assert hw["dispatch_id"] == dispatch_dir.name
        assert hw["fingerprint"].startswith("gpu:")
        assert isinstance(hw["hosts"], dict)


def test_unsupported_pair_lands_in_skipped_array(tmp_path: Path, stub_ssh_runner, stub_provisioner):
    """End-to-end: a (task, backend) the task doesn't support → skipped[]."""
    if not _ssh_localhost_works():
        pytest.skip("ssh localhost does not work without a password; skipping integration test")

    # Build a physx yaml with one supported and one unsupported task.
    el = EnvList()
    el.groups["direct/ant"] = [
        EnvEntry(
            task_id="Isaac-Ant-Direct-v0",
            entry_point="ep:E",
            env_cfg_entry_point="ec:E",
            group="direct/ant",
            has_rsl_rl=True,
            has_skrl=True,
            framework="rsl_rl",
            num_envs=4096,
            max_iterations=10,
            keep=True,
            presets_available=["physx", "newton"],
        ),
        EnvEntry(
            task_id="Isaac-NewtonOnly-v0",
            entry_point="ep:N",
            env_cfg_entry_point="ec:N",
            group="direct/ant",
            has_rsl_rl=True,
            has_skrl=True,
            framework="rsl_rl",
            num_envs=4096,
            max_iterations=10,
            keep=True,
            presets_available=["newton"],
        ),
    ]
    physx_yaml = tmp_path / "physx.yaml"
    write_env_list(physx_yaml, el, generator="test")

    # One-host fleet targeting localhost; same loopback shape as the
    # success-path test above.
    repo_root = Path.cwd()
    host = ValkyrieConfig(
        host="localhost",
        ssh_user=os.environ.get("USER", "root"),
        isaaclab_path=str(repo_root),
    )
    fleet = Fleet(fleet_name="loopback-test", hosts=[host])
    dispatch_dir = tmp_path / "odin_runs" / "20260427-140000"
    dispatch_dir.mkdir(parents=True)

    state = run_dispatch(
        fleet=fleet,
        physx_yaml=physx_yaml,
        newton_yaml=None,
        dispatch_dir=dispatch_dir,
        options=DispatchOptions(seeds=[42, 43], per_job_timeout_s=60, skip_aggregate=True, detached_mode=False),
        ssh=ShellSSHRunner(),
        rsync=ShellRsyncRunner(),
    )

    # Ant ran (2 seeds); NewtonOnly skipped (2 seeds).
    assert {j.task_id for j in state.jobs} == {"Isaac-Ant-Direct-v0"}
    assert len(state.jobs) == 2
    assert {sk.task_id for sk in state.skipped} == {"Isaac-NewtonOnly-v0"}
    assert len(state.skipped) == 2
    assert all(sk.reason == "preset_unsupported" for sk in state.skipped)
    assert all(sk.backend == "physx" for sk in state.skipped)
    assert {sk.seed for sk in state.skipped} == {42, 43}

    # And dispatch.json on disk reflects the same.
    from tools.odin.asgard.state import read_dispatch_state

    reloaded = read_dispatch_state(dispatch_dir)
    assert reloaded is not None
    assert len(reloaded.skipped) == 2
    assert reloaded.skipped[0].presets_available == ["newton"]


def test_native_match_runs_unsupported_pair_routes_to_skipped(tmp_path: Path, stub_ssh_runner, stub_provisioner):
    """End-to-end: presets_available=[] AND native_backend != requested → skipped[]."""
    if not _ssh_localhost_works():
        pytest.skip("ssh localhost not configured")

    el = EnvList()
    el.groups["direct/quadcopter"] = [
        EnvEntry(
            task_id="Isaac-Quadcopter-Direct-v0",
            entry_point="ep:E",
            env_cfg_entry_point="ec:E",
            group="direct/quadcopter",
            has_rsl_rl=True,
            has_skrl=True,
            framework="rsl_rl",
            num_envs=4096,
            max_iterations=10,
            keep=True,
            presets_available=[],
            native_backend="physx",
        ),
    ]
    yaml_path = tmp_path / "envs.yaml"
    write_env_list(yaml_path, el, generator="test")

    dispatch_dir = tmp_path / "20260427-160000"
    dispatch_dir.mkdir()
    fleet = Fleet(
        fleet_name="loopback-test",
        hosts=[
            ValkyrieConfig(
                host="localhost",
                ssh_user=os.environ.get("USER") or "root",
                ssh_key=None,
                isaaclab_path=str(tmp_path / "remote_isaaclab"),
                container_name="loopback-container",
            ),
        ],
    )
    state = run_dispatch(
        fleet=fleet,
        physx_yaml=None,
        newton_yaml=yaml_path,  # request newton on a physx-native task
        dispatch_dir=dispatch_dir,
        options=DispatchOptions(seeds=[42, 43], skip_aggregate=True, per_job_timeout_s=60, detached_mode=False),
        ssh=ShellSSHRunner(),
        rsync=ShellRsyncRunner(),
    )

    # Quadcopter never queued; landed in skipped[] with the new reason.
    assert state.jobs == []
    assert len(state.skipped) == 2
    assert {sk.task_id for sk in state.skipped} == {"Isaac-Quadcopter-Direct-v0"}
    assert all(sk.reason == "native_backend_mismatch" for sk in state.skipped)
    assert all(sk.backend == "newton" for sk in state.skipped)
    assert all(sk.native_backend == "physx" for sk in state.skipped)
    assert {sk.seed for sk in state.skipped} == {42, 43}

    # And dispatch.json on disk reflects the same.
    from tools.odin.asgard.state import read_dispatch_state

    reloaded = read_dispatch_state(dispatch_dir)
    assert reloaded is not None
    assert reloaded.schema_version == "1.4"
    assert len(reloaded.skipped) == 2
    assert reloaded.skipped[0].native_backend == "physx"


@pytest.fixture
def stub_ssh_runner_first_job_nvml(monkeypatch, tmp_path: Path):
    """Like ``stub_ssh_runner``, but the FIRST docker-exec call per host fails
    with NVML stderr; subsequent calls succeed and materialise a valid bundle.

    Yields the ``seen_per_host`` counter so the test can assert that the second
    attempt landed on the same host after a successful recovery.
    """
    from tools.odin.asgard import worker as worker_mod

    real_build = worker_mod._build_docker_exec_cmd
    seen_per_host: dict[str, int] = {}

    def _fake_build(host: ValkyrieConfig, job) -> str:
        seen = seen_per_host.get(host.host, 0)
        seen_per_host[host.host] = seen + 1
        if seen == 0:
            # First call on this host: emit a recognised NVML signature on stderr
            # and exit non-zero. Worker._classify maps this to gpu_lost.
            return "echo 'Failed to initialize NVML: Unknown Error' 1>&2 && exit 1"
        # Subsequent calls: materialise a valid bundle and exit 0.
        bundle_dir = f"{host.isaaclab_path}/odin_runs/{job.bundle_dir_name}"
        manifest = {
            "schema_version": "1.0",
            "phases": {"training": {"status": "completed"}, "startup": {"status": "completed"}},
        }
        training = {"schema_version": "1.0"}
        startup = {"schema_version": "1.0"}
        manifest_s = json.dumps(manifest).replace("'", r"\'")
        training_s = json.dumps(training).replace("'", r"\'")
        startup_s = json.dumps(startup).replace("'", r"\'")
        return (
            f"mkdir -p {bundle_dir} && "
            f"printf '%s' '{manifest_s}' > {bundle_dir}/manifest.json && "
            f"printf '%s' '{training_s}' > {bundle_dir}/training.json && "
            f"printf '%s' '{startup_s}' > {bundle_dir}/startup.json"
        )

    monkeypatch.setattr(worker_mod, "_build_docker_exec_cmd", _fake_build)
    yield seen_per_host
    monkeypatch.setattr(worker_mod, "_build_docker_exec_cmd", real_build)


def test_loopback_dispatch_recovers_from_gpu_lost(
    tmp_path: Path,
    stub_ssh_runner_first_job_nvml,
    stub_provisioner,
    monkeypatch,
):
    """End-to-end: host's first job emits NVML stderr → ``_classify`` flags
    ``gpu_lost`` → fake ``recover_valkyrie_gpu`` returns ``recovered=True`` →
    second attempt completes the job. Final ``dispatch.json`` shows job
    completed, attempts==2, and ``fleet[host].last_error == 'gpu_lost: recovered'``.
    """
    if not _ssh_localhost_works():
        pytest.skip("ssh localhost does not work without a password; skipping integration test")

    from tools.odin.asgard import worker as worker_mod
    from tools.odin.asgard.recovery import RecoveryResult

    recover_calls: list[str] = []

    def _fake_recover(host, *, ssh):
        recover_calls.append(host.host)
        return RecoveryResult(
            host=host.host,
            container_name=host.container_name,
            attempted=True,
            recovered=True,
            duration_s=10.0,
            message="recovered_via_container_restart",
            details={},
        )

    monkeypatch.setattr(worker_mod, "recover_valkyrie_gpu", _fake_recover)

    # Build a one-row env list (mirror of the success-path test above).
    el = EnvList()
    el.groups["direct/ant"] = [
        EnvEntry(
            task_id="Isaac-Ant-Direct-v0",
            entry_point="isaaclab_tasks.direct.ant:AntEnv",
            env_cfg_entry_point="isaaclab_tasks.direct.ant.ant_env_cfg:AntEnvCfg",
            group="direct/ant",
            has_rsl_rl=True,
            has_skrl=True,
            has_rl_games=False,
            framework="rsl_rl",
            num_envs=4096,
            max_iterations=10,
            keep=True,
            status="current",
        )
    ]
    physx_yaml = tmp_path / "physx.yaml"
    write_env_list(physx_yaml, el, generator="test")

    repo_root = Path.cwd()
    host = ValkyrieConfig(
        host="localhost",
        ssh_user=os.environ.get("USER", "root"),
        isaaclab_path=str(repo_root),
    )
    fleet = Fleet(fleet_name="loopback-recovery", hosts=[host])
    dispatch_dir = tmp_path / "odin_runs" / "20260427-recover"
    dispatch_dir.mkdir(parents=True)

    state = run_dispatch(
        fleet=fleet,
        physx_yaml=physx_yaml,
        newton_yaml=None,
        dispatch_dir=dispatch_dir,
        options=DispatchOptions(seeds=[42], per_job_timeout_s=60, skip_aggregate=True, detached_mode=False),
        ssh=ShellSSHRunner(),
        rsync=ShellRsyncRunner(),
    )

    # The job recovered and completed on its second attempt.
    assert len(state.jobs) == 1
    assert state.jobs[0].status == "completed", f"job failed: {state.jobs[0].failure}"
    assert state.jobs[0].attempts == 2
    fleet_entry = next(f for f in state.fleet if f.host == "localhost")
    assert fleet_entry.last_error == "gpu_lost: recovered"
    assert recover_calls == ["localhost"]

    # On-disk dispatch.json mirrors the in-memory state.
    dj = json.loads((dispatch_dir / "dispatch.json").read_text())
    assert dj["jobs"][0]["status"] == "completed"
    assert dj["jobs"][0]["attempts"] == 2
    on_disk_fleet = next(f for f in dj["fleet"] if f["host"] == "localhost")
    assert on_disk_fleet["last_error"] == "gpu_lost: recovered"

    # Bundle was pulled back on the recovered attempt.
    bundle = dispatch_dir / state.jobs[0].bundle_dir_name
    assert (bundle / "manifest.json").exists()


# --- Detached-mode loopback resume ------------------------------------------


@pytest.fixture
def stub_detached_runner(monkeypatch):
    """Replace ``_build_submit_script`` and ``_build_poll_script`` with local
    stubs that bypass docker exec / isaacsim.

    The submit stub materialises a fake bundle directly on the host's
    filesystem (which is the same disk as the dispatcher in the
    localhost loopback). The poll stub then runs the same bash test
    logic against the real (un-prefixed) host path, so the worker's
    ``_finalize_terminal`` sees ``done`` and finalizes correctly.
    """
    from tools.odin.asgard import worker as worker_mod

    real_submit = worker_mod._build_submit_script
    real_poll = worker_mod._build_poll_script

    def _fake_submit(host, job, *, submitted_at, per_job_timeout_s):
        bundle_dir = f"{host.isaaclab_path}/odin_runs/{job.bundle_dir_name}"
        manifest = {
            "schema_version": "1.0",
            "phases": {"training": {"status": "completed"}, "startup": {"status": "completed"}},
        }
        training = {"schema_version": "1.0"}
        startup = {"schema_version": "1.0"}
        manifest_s = json.dumps(manifest).replace("'", r"\'")
        training_s = json.dumps(training).replace("'", r"\'")
        startup_s = json.dumps(startup).replace("'", r"\'")
        return (
            f"mkdir -p {bundle_dir}/logs && "
            f"echo $$ > {bundle_dir}/.run.pid && "
            f"printf '%s' '{manifest_s}' > {bundle_dir}/manifest.json && "
            f"printf '%s' '{training_s}' > {bundle_dir}/training.json && "
            f"printf '%s' '{startup_s}' > {bundle_dir}/startup.json && "
            f"echo 'odin-submit: ok run_id={job.run_id} bundle={job.bundle_dir_name}'"
        )

    def _fake_poll(host, bundle_ids):
        bundles = " ".join(bundle_ids)
        # Same logic as real poll, but rooted at host.isaaclab_path
        # instead of /workspace/isaaclab and without docker exec.
        inner = (
            f"for bundle in {bundles}; do "
            f"if [ -f {host.isaaclab_path}/odin_runs/$bundle/manifest.json ]; then "
            f'echo "$bundle done"; '
            f"elif [ -f {host.isaaclab_path}/odin_runs/$bundle/.run.pid ]; then "
            f"pid=$(cat {host.isaaclab_path}/odin_runs/$bundle/.run.pid); "
            f'if kill -0 "$pid" 2>/dev/null; then echo "$bundle alive"; '
            f'else echo "$bundle exited-no-manifest"; fi; '
            f'else echo "$bundle no-pidfile"; fi; '
            f"done"
        )
        return inner

    monkeypatch.setattr(worker_mod, "_build_submit_script", _fake_submit)
    monkeypatch.setattr(worker_mod, "_build_poll_script", _fake_poll)
    yield
    monkeypatch.setattr(worker_mod, "_build_submit_script", real_submit)
    monkeypatch.setattr(worker_mod, "_build_poll_script", real_poll)


def test_loopback_detached_dispatch_completes(tmp_path: Path, stub_detached_runner, stub_provisioner):
    """End-to-end with detached_mode=True: submit → poll sees manifest →
    rsync pull → completed.

    Fast smoke test for the new path; the longer dispatcher-restart
    scenario is a separate test that needs real backgrounding.
    """
    if not _ssh_localhost_works():
        pytest.skip("ssh localhost does not work without a password; skipping integration test")

    from tools.odin.common.env_list import EnvEntry as _EnvEntry
    from tools.odin.common.env_list import EnvList as _EnvList
    from tools.odin.common.env_list import write_env_list as _write_env_list

    el = _EnvList()
    el.groups["direct/ant"] = [
        _EnvEntry(
            task_id="Isaac-Ant-Direct-v0",
            entry_point="ep:E",
            env_cfg_entry_point="ec:E",
            group="direct/ant",
            has_rsl_rl=True,
            has_skrl=True,
            framework="rsl_rl",
            num_envs=4096,
            max_iterations=10,
            keep=True,
            status="current",
        )
    ]
    physx_yaml = tmp_path / "physx.yaml"
    _write_env_list(physx_yaml, el, generator="test")

    repo_root = Path.cwd()
    host = ValkyrieConfig(
        host="localhost",
        ssh_user=os.environ.get("USER", "root"),
        isaaclab_path=str(repo_root),
    )
    fleet = Fleet(fleet_name="loopback-detached", hosts=[host])
    dispatch_dir = tmp_path / "odin_runs" / "20260430-detached"
    dispatch_dir.mkdir(parents=True)

    state = run_dispatch(
        fleet=fleet,
        physx_yaml=physx_yaml,
        newton_yaml=None,
        dispatch_dir=dispatch_dir,
        options=DispatchOptions(
            seeds=[42],
            per_job_timeout_s=60,
            skip_aggregate=True,
            detached_mode=True,
            poll_interval_s=0,  # tighten the run loop for the test
        ),
        ssh=ShellSSHRunner(),
        rsync=ShellRsyncRunner(),
    )

    # The job submitted, was polled, hit POLL_DONE, and finalized as completed.
    assert len(state.jobs) == 1
    assert state.jobs[0].status == "completed", f"job failed: {state.jobs[0].failure}"
    bundle = dispatch_dir / state.jobs[0].bundle_dir_name
    assert (bundle / "manifest.json").exists()


def test_loopback_detached_resume_reattaches_inflight(tmp_path: Path, stub_provisioner, monkeypatch):
    """Dispatcher restart: a prior run left a job in 'running' on the remote
    with a live trainer. ``--resume`` reattaches via reconcile and finalizes.

    Setup mimics what a real dispatcher crash would leave behind: the
    bundle dir has a ``.tracker.json`` and ``.run.pid``, the trainer
    "process" is still running (we use ``sleep`` as the stand-in), and
    no ``manifest.json`` is present yet. The resumed dispatcher should
    poll → see ``alive`` first, then once the sleep completes and the
    fixture creates the manifest, → ``done`` → ``completed``.
    """
    if not _ssh_localhost_works():
        pytest.skip("ssh localhost does not work without a password; skipping integration test")

    from tools.odin.asgard import reconcile as reconcile_mod
    from tools.odin.asgard import worker as worker_mod
    from tools.odin.asgard.jobs import JobEntry
    from tools.odin.asgard.state import (
        SCHEMA_VERSION,
        DispatchState,
        FleetSnapshot,
        write_dispatch_state,
    )
    from tools.odin.asgard.tracker import Tracker, write_tracker
    from tools.odin.common.env_list import EnvEntry as _EnvEntry
    from tools.odin.common.env_list import EnvList as _EnvList
    from tools.odin.common.env_list import write_env_list as _write_env_list

    # Same poll stub as the smoke test (path-rooted, no docker).
    def _fake_poll(host, bundle_ids):
        bundles = " ".join(bundle_ids)
        return (
            f"for bundle in {bundles}; do "
            f"if [ -f {host.isaaclab_path}/odin_runs/$bundle/manifest.json ]; then "
            f'echo "$bundle done"; '
            f"elif [ -f {host.isaaclab_path}/odin_runs/$bundle/.run.pid ]; then "
            f"pid=$(cat {host.isaaclab_path}/odin_runs/$bundle/.run.pid); "
            f'if kill -0 "$pid" 2>/dev/null; then echo "$bundle alive"; '
            f'else echo "$bundle exited-no-manifest"; fi; '
            f'else echo "$bundle no-pidfile"; fi; '
            f"done"
        )

    monkeypatch.setattr(worker_mod, "_build_poll_script", _fake_poll)
    # Reconcile uses the same import-time symbol; patching the worker
    # module is enough since reconcile imports it at call time.

    # Manifest cat for reconcile's first probe — initially missing, so
    # reconcile falls into the detached poll path.
    real_read_manifest = reconcile_mod._read_remote_manifest

    def _fake_read_manifest(host, run_id, ssh):
        return None  # always "no manifest" for this test

    monkeypatch.setattr(reconcile_mod, "_read_remote_manifest", _fake_read_manifest)

    # Build the env list + per-host config.
    el = _EnvList()
    el.groups["direct/ant"] = [
        _EnvEntry(
            task_id="Isaac-Ant-Direct-v0",
            entry_point="ep:E",
            env_cfg_entry_point="ec:E",
            group="direct/ant",
            has_rsl_rl=True,
            has_skrl=True,
            framework="rsl_rl",
            num_envs=4096,
            max_iterations=10,
            keep=True,
            status="current",
        )
    ]
    physx_yaml = tmp_path / "physx.yaml"
    _write_env_list(physx_yaml, el, generator="test")

    repo_root = Path.cwd()
    host = ValkyrieConfig(
        host="localhost",
        ssh_user=os.environ.get("USER", "root"),
        isaaclab_path=str(repo_root),
    )
    fleet = Fleet(fleet_name="loopback-resume", hosts=[host])
    dispatch_dir = tmp_path / "odin_runs" / "20260430-resume"
    dispatch_dir.mkdir(parents=True)

    # Stage prior dispatch.json with a single 'running' job.
    run_id = "rsl-rl_physx_Isaac-Ant-Direct-v0_20260430-resume_seed42"
    job = JobEntry(
        run_id=run_id,
        task_id="Isaac-Ant-Direct-v0",
        framework="rsl_rl",
        backend="physx",
        num_envs=4096,
        max_iterations=10,
        seed=42,
        bundle_dir_name=run_id,
        status="running",
        assigned_to="localhost",
        attempts=1,
        started_at="2026-04-30T11:00:00Z",
    )
    prior = DispatchState(
        schema_version=SCHEMA_VERSION,
        dispatch_id=dispatch_dir.name,
        started_at="2026-04-30T11:00:00Z",
        ended_at=None,
        seeds=[42],
        commit_sha="resume-test",
        fleet=[FleetSnapshot(host="localhost", status="busy", current_run_id=run_id)],
        jobs=[job],
    )
    write_dispatch_state(dispatch_dir, prior)

    # Stage the remote bundle with a live "trainer" (background sleep) +
    # tracker + pidfile, no manifest yet.
    remote_bundle = repo_root / "odin_runs" / run_id
    remote_bundle.mkdir(parents=True, exist_ok=True)
    (remote_bundle / "logs").mkdir(parents=True, exist_ok=True)
    write_tracker(
        remote_bundle,
        Tracker(
            run_id=run_id,
            container_name="isaac-lab-base",
            host="localhost",
            submitted_at="2026-04-30T11:00:00Z",
            pid=os.getpid(),  # any live PID works for kill -0
            per_job_timeout_s=43200,
        ),
    )
    (remote_bundle / ".run.pid").write_text(f"{os.getpid()}\n")
    # Pre-write the manifest so the reattached worker's first poll sees
    # "done" and finalizes immediately. (Mimics the trainer finishing
    # while the dispatcher was down.)
    (remote_bundle / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "phases": {"startup": {"status": "completed"}, "training": {"status": "completed"}},
            }
        )
    )
    (remote_bundle / "training.json").write_text(json.dumps({"schema_version": "1.0"}))
    (remote_bundle / "startup.json").write_text(json.dumps({"schema_version": "1.0"}))

    try:
        state = run_dispatch(
            fleet=fleet,
            physx_yaml=physx_yaml,
            newton_yaml=None,
            dispatch_dir=dispatch_dir,
            options=DispatchOptions(
                seeds=[42],
                per_job_timeout_s=60,
                skip_aggregate=True,
                detached_mode=True,
                poll_interval_s=0,
            ),
            ssh=ShellSSHRunner(),
            rsync=ShellRsyncRunner(),
        )
    finally:
        # Clean up the staged remote bundle so re-running the test isn't
        # confused by stale files.
        import shutil

        shutil.rmtree(remote_bundle, ignore_errors=True)
        monkeypatch.setattr(reconcile_mod, "_read_remote_manifest", real_read_manifest)

    # The resumed worker reattached, polled → done, finalized → completed.
    assert len(state.jobs) == 1
    assert state.jobs[0].status == "completed", f"job failed: {state.jobs[0].failure}"


def test_loopback_detached_dispatch_skip_and_kill_via_db(tmp_path: Path, stub_provisioner, monkeypatch):
    """Two-job loopback dispatch: skip one before submit, kill the other mid-run.

    Asserts both end terminal with the expected kinds and the killed job's
    bundle dir contains its (partial) logs/.
    """
    if not _ssh_localhost_works():
        pytest.skip("ssh localhost does not work without a password")

    from tools.odin.asgard import worker as worker_mod
    from tools.odin.valhalla.dashboard.cancel_db import CancelDB

    # Submit stub: write a tracker + pidfile (using the test process's pid so
    # poll's `kill -0` reports `alive`), then sleep so the job is observably
    # mid-flight when we issue the kill.
    test_pid = os.getpid()

    def _fake_submit(host, job, *, submitted_at, per_job_timeout_s):
        bundle_dir = f"{host.isaaclab_path}/odin_runs/{job.bundle_dir_name}"
        import json as _json

        from tools.odin.asgard.tracker import TRACKER_SCHEMA_VERSION

        tracker = {
            "schema_version": TRACKER_SCHEMA_VERSION,
            "run_id": job.run_id,
            "container_name": host.container_name,
            "host": host.host,
            "submitted_at": submitted_at,
            "pid": test_pid,
            "container_pid": None,
            "per_job_timeout_s": per_job_timeout_s,
        }
        tracker_s = _json.dumps(tracker).replace("'", r"\'")
        return (
            f"mkdir -p {bundle_dir}/logs && "
            # Write the test process's PID so `kill -0` always reports alive.
            f"echo {test_pid} > {bundle_dir}/.run.pid && "
            f"printf '%s' '{tracker_s}' > {bundle_dir}/.tracker.json && "
            # Simulate a mid-run trainer with partial stderr.
            f"echo 'fake training started' > {bundle_dir}/logs/hugin-stderr.log && "
            f"echo 'odin-submit: ok run_id={job.run_id} bundle={job.bundle_dir_name}'"
        )

    def _fake_poll(host, bundle_ids):
        bundles = " ".join(bundle_ids)
        return (
            f"for bundle in {bundles}; do "
            f"if [ -f {host.isaaclab_path}/odin_runs/$bundle/manifest.json ]; then "
            f'echo "$bundle done"; '
            f"elif [ -f {host.isaaclab_path}/odin_runs/$bundle/.run.pid ]; then "
            f"pid=$(cat {host.isaaclab_path}/odin_runs/$bundle/.run.pid); "
            f'if kill -0 "$pid" 2>/dev/null; then echo "$bundle alive"; '
            f'else echo "$bundle exited-no-manifest"; fi; '
            f'else echo "$bundle no-pidfile"; fi; '
            f"done"
        )

    monkeypatch.setattr(worker_mod, "_build_submit_script", _fake_submit)
    monkeypatch.setattr(worker_mod, "_build_poll_script", _fake_poll)

    # Stub _cleanup_remote_process to simulate killing the remote process:
    # replace .run.pid with a dead PID so the next poll returns
    # "exited-no-manifest" (pidfile present but `kill -0` fails) rather than
    # "no-pidfile".  The real implementation issues docker-exec pkill, which
    # cannot reach our test-process PID in a loopback test.
    def _fake_cleanup(self_worker, job):
        pid_file = Path(self_worker.host.isaaclab_path) / "odin_runs" / job.bundle_dir_name / ".run.pid"
        # PID 2147483647 (INT_MAX) is virtually never a live process.
        pid_file.write_text("2147483647\n")

    monkeypatch.setattr(worker_mod.ValkyrieWorker, "_cleanup_remote_process", _fake_cleanup)

    repo_root = Path.cwd()
    host = ValkyrieConfig(
        host="localhost",
        ssh_user=os.environ.get("USER", "root"),
        isaaclab_path=str(repo_root),
    )
    fleet = Fleet(fleet_name="loopback-cancel", hosts=[host])
    dispatch_dir = tmp_path / "odin_runs" / "20260504-cancel"
    dispatch_dir.mkdir(parents=True)

    # Two-job env list (we need a pending one to skip + a running one to kill).
    from tools.odin.common.env_list import EnvEntry as _EnvEntry
    from tools.odin.common.env_list import EnvList as _EnvList
    from tools.odin.common.env_list import write_env_list as _write_env_list

    el = _EnvList()
    el.groups["direct/ant"] = [
        _EnvEntry(
            task_id="Isaac-Ant-Direct-v0",
            entry_point="ep:E",
            env_cfg_entry_point="ec:E",
            group="direct/ant",
            has_rsl_rl=True,
            has_skrl=True,
            framework="rsl_rl",
            num_envs=4096,
            max_iterations=10,
            keep=True,
            status="current",
        )
    ]
    physx_yaml = tmp_path / "physx.yaml"
    _write_env_list(physx_yaml, el, generator="test")

    # Kick the dispatch in the background; it will spin polling forever
    # because we never write a manifest. We send the cancel rows after a
    # short sleep, then expect the dispatch to terminate.
    cancel_db = CancelDB(dispatch_dir.parent)

    import threading as _threading

    def _send_cancels():
        import time as _time

        _time.sleep(2.0)  # let the runner submit + poll at least once
        # Job ids are deterministic from dispatch_id + framework + task + seed.
        # seed42 is the first job submitted → running → send kill.
        # seed43 is still pending in the queue → send skip (lands terminal
        # synchronously so `remaining` reaches 0 quickly).  If seed43 is
        # somehow already running, _consume_cancellations upgrades skip→kill.
        dispatch_id = dispatch_dir.name
        run_id_42 = f"rsl-rl_physx_Isaac-Ant-Direct-v0_{dispatch_id}_seed42"
        run_id_43 = f"rsl-rl_physx_Isaac-Ant-Direct-v0_{dispatch_id}_seed43"
        cancel_db.request(dispatch_id, run_id_42, kind="kill")
        cancel_db.request(dispatch_id, run_id_43, kind="skip")

    cancel_thread = _threading.Thread(target=_send_cancels, daemon=True)
    cancel_thread.start()

    state = run_dispatch(
        fleet=fleet,
        physx_yaml=physx_yaml,
        newton_yaml=None,
        dispatch_dir=dispatch_dir,
        options=DispatchOptions(
            seeds=[42, 43],  # two jobs from one task
            per_job_timeout_s=60,
            skip_aggregate=True,
            detached_mode=True,
            poll_interval_s=0,
            live_retry_poll_s=0.5,
        ),
        ssh=ShellSSHRunner(),
        rsync=ShellRsyncRunner(),
    )

    # Pre-cleanup: the loopback wrote bundles into repo_root.
    import shutil as _shutil

    for j in state.jobs:
        _shutil.rmtree(repo_root / "odin_runs" / j.bundle_dir_name, ignore_errors=True)

    # Whichever job the runner sent first becomes "running" + killed; the
    # other stays pending until killed via the same cancel-loop tick or
    # through the per-job timeout. In a deterministic test we only assert
    # on the killed one.
    killed_jobs = [j for j in state.jobs if j.failure and j.failure.kind == "killed"]
    assert killed_jobs, (
        f"expected at least one killed job, got "
        f"{[(j.run_id, j.failure.kind if j.failure else j.status) for j in state.jobs]}"
    )
    killed = killed_jobs[0]
    bundle = dispatch_dir / killed.bundle_dir_name
    # Partial logs preserved per kill-flow spec.
    assert (bundle / "logs" / "hugin-stderr.log").exists()


def test_loopback_dispatch_recovers_from_synchronous_gpu_lost(
    tmp_path: Path,
    stub_provisioner,
    monkeypatch,
):
    """Bug 3 end-to-end regression: a pre-submit GPU probe failure
    triggers worker._handle_synchronous_failure(gpu_lost), recovery
    succeeds, and the job re-runs successfully on the second attempt
    with status='completed' in the final dispatch.json. Crucially:
    the JobEntry must NOT spend any time stuck at status='running'
    after the synchronous failure — the first call site that sees
    the JobEntry post-recovery must observe status='pending'.

    This test exercises the detached-mode submit path (pty=False), where
    the remote stderr is captured as a real pipe and the
    ``odin: gpu_unavailable`` marker can be matched by ``_submit_job``'s
    GPU-loss check before falling through to ``_handle_synchronous_failure``.
    """
    if not _ssh_localhost_works():
        pytest.skip("ssh localhost does not work without a password; skipping integration test")

    from tools.odin.asgard import worker as worker_mod
    from tools.odin.asgard.recovery import RecoveryResult

    # Track how many submit attempts have been made per host so we can flip
    # from failure to success on the second call.
    seen_per_host: dict[str, int] = {}

    def _fake_submit(host, job, *, submitted_at, per_job_timeout_s):
        n = seen_per_host.get(host.host, 0)
        seen_per_host[host.host] = n + 1
        if n == 0:
            # First submit: synthesize a pre-submit GPU probe failure.
            # Emitting the 'odin: gpu_unavailable: ...' marker on stderr with
            # exit 1 is exactly what the real nvidia-smi probe section of
            # _build_submit_script does when nvidia-smi returns non-zero.
            # _submit_job's GPU-loss branch sees this and returns a gpu_lost
            # FailureInfo → _handle_synchronous_failure → recovery → re-queue.
            return "echo 'odin: gpu_unavailable: nvidia-smi -L failed' 1>&2 && exit 1"
        # Second submit: write a minimal valid bundle and echo the sentinel so
        # _submit_job treats it as success (exit 0 + odin-submit: ok).
        bundle_dir = f"{host.isaaclab_path}/odin_runs/{job.bundle_dir_name}"
        manifest = {
            "schema_version": "1.0",
            "phases": {"startup": {"status": "completed"}, "training": {"status": "completed"}},
        }
        training = {"schema_version": "1.0"}
        startup = {"schema_version": "1.0"}
        manifest_s = json.dumps(manifest).replace("'", r"\'")
        training_s = json.dumps(training).replace("'", r"\'")
        startup_s = json.dumps(startup).replace("'", r"\'")
        return (
            f"mkdir -p {bundle_dir}/logs && "
            f"echo $$ > {bundle_dir}/.run.pid && "
            f"printf '%s' '{manifest_s}' > {bundle_dir}/manifest.json && "
            f"printf '%s' '{training_s}' > {bundle_dir}/training.json && "
            f"printf '%s' '{startup_s}' > {bundle_dir}/startup.json && "
            f"echo 'odin-submit: ok run_id={job.run_id} bundle={job.bundle_dir_name}'"
        )

    def _fake_poll(host, bundle_ids):
        bundles = " ".join(bundle_ids)
        return (
            f"for bundle in {bundles}; do "
            f"if [ -f {host.isaaclab_path}/odin_runs/$bundle/manifest.json ]; then "
            f'echo "$bundle done"; '
            f"elif [ -f {host.isaaclab_path}/odin_runs/$bundle/.run.pid ]; then "
            f"pid=$(cat {host.isaaclab_path}/odin_runs/$bundle/.run.pid); "
            f'if kill -0 "$pid" 2>/dev/null; then echo "$bundle alive"; '
            f'else echo "$bundle exited-no-manifest"; fi; '
            f'else echo "$bundle no-pidfile"; fi; '
            f"done"
        )

    monkeypatch.setattr(worker_mod, "_build_submit_script", _fake_submit)
    monkeypatch.setattr(worker_mod, "_build_poll_script", _fake_poll)

    # Mock GPU recovery to succeed immediately.
    monkeypatch.setattr(
        worker_mod,
        "recover_valkyrie_gpu",
        lambda host, *, ssh: RecoveryResult(
            host=host.host,
            container_name=host.container_name,
            attempted=True,
            recovered=True,
            duration_s=1.0,
            message="ok",
        ),
    )

    # Build a one-row env list (mirrors test_loopback_dispatch_recovers_from_gpu_lost).
    el = EnvList()
    el.groups["direct/ant"] = [
        EnvEntry(
            task_id="Isaac-Ant-Direct-v0",
            entry_point="isaaclab_tasks.direct.ant:AntEnv",
            env_cfg_entry_point="isaaclab_tasks.direct.ant.ant_env_cfg:AntEnvCfg",
            group="direct/ant",
            has_rsl_rl=True,
            has_skrl=True,
            has_rl_games=False,
            framework="rsl_rl",
            num_envs=4096,
            max_iterations=10,
            keep=True,
            status="current",
        )
    ]
    physx_yaml = tmp_path / "physx.yaml"
    write_env_list(physx_yaml, el, generator="test")

    repo_root = Path.cwd()
    host = ValkyrieConfig(
        host="localhost",
        ssh_user=os.environ.get("USER", "root"),
        isaaclab_path=str(repo_root),
    )
    fleet = Fleet(fleet_name="loopback-sync-gpu-recovery", hosts=[host])
    dispatch_dir = tmp_path / "odin_runs" / "20260504-syncgpu"
    dispatch_dir.mkdir(parents=True)

    state = run_dispatch(
        fleet=fleet,
        physx_yaml=physx_yaml,
        newton_yaml=None,
        dispatch_dir=dispatch_dir,
        options=DispatchOptions(
            seeds=[42],
            per_job_timeout_s=60,
            skip_aggregate=True,
            detached_mode=True,
            poll_interval_s=0,
        ),
        ssh=ShellSSHRunner(),
        rsync=ShellRsyncRunner(),
    )

    # The job must have recovered and completed on its second attempt.
    assert len(state.jobs) == 1
    assert state.jobs[0].status == "completed", f"job failed: {state.jobs[0].failure}"
    assert state.jobs[0].ended_at is not None  # Bug 4 invariant: always set on terminal
    assert state.jobs[0].attempts == 2  # one failed synchronous submit, one successful

    # On-disk dispatch.json mirrors the in-memory state.
    dj = json.loads((dispatch_dir / "dispatch.json").read_text())
    assert dj["jobs"][0]["status"] == "completed"
    assert dj["jobs"][0]["attempts"] == 2

    # Bundle was pulled back on the recovered attempt.
    bundle = dispatch_dir / state.jobs[0].bundle_dir_name
    assert (bundle / "manifest.json").exists()
