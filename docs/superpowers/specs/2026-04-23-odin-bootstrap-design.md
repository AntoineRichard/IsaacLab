# Odin Bootstrap — Design

**Status:** approved
**Date:** 2026-04-23
**Task covered:** Close a real T3.1 gap: bringing a fresh Valkyrie from
"bare machine with SSH + Docker + GPU" to "T3.1-preflight-ready" state.
Implements a new `odin-bootstrap` CLI + a small provisioner tweak.

## 1. Motivation

T3.1's `run_dispatch` assumes each Valkyrie in `fleet.yaml` is already
preflight-ready: the IsaacLab repo is cloned at `isaaclab_path`, the
named docker container is `running`, and the Docker daemon + SSH are
up. In practice, when a new fleet is plugged in (as we just discovered
on 10.176.221.98 / 10.63.172.46), the machines have only:

- SSH access,
- Docker installed,
- an NVIDIA GPU,

and nothing else. T3.1's preflight fails on every such host because
the `container_up` and `isaaclab_present` checks run **before** any
provisioning step. `run_dispatch` then aborts with `preflight failed
for all N hosts` before the provisioner has a chance to create either
artifact.

Worse: even if we bypassed preflight, the provisioner's
`_container_start` hardcodes a 300 s timeout. A first-time
`./docker/container.py start` pulls the base Isaac Lab image + builds
on top, which typically takes 15-30 minutes. The 300 s timeout was
correct for the warm path (subsequent dispatches against already-built
containers) but is too short for the cold path.

This spec adds the missing "first contact" pathway: a dedicated
`odin-bootstrap` command + a minimal provisioner tweak, leaving the
T3.1 dispatch path unchanged.

## 2. Goals

- A fresh Valkyrie (just SSH + Docker + GPU) can be brought to
  T3.1-preflight-ready state with one command:
  `odin-bootstrap --fleet fleet.yaml`.
- Per-host behavior: wipe any prior IsaacLab tree, rsync the
  controller's working tree, run `./docker/container.py start` with a
  long enough timeout to accommodate a first-time image build, then
  verify the container is `running`.
- Idempotent and rerunnable: bootstrapping an already-bootstrapped
  host must succeed (the wipe + re-rsync cycle is always safe).
- Parallel across hosts (one thread per Valkyrie), same concurrency
  pattern as T3.1's dispatch workers.
- Informative: per-host per-step timings, per-host overall
  `ok` / `message` / `commit_sha`, plus a concise summary line.

## 3. Non-goals

- No changes to T3.1's `run_dispatch`, preflight, worker, or
  dispatch state-machine. Once bootstrap lands a host at preflight-
  ready, T3.1 drives it normally.
- No new schema files. The bootstrap command writes zero JSON state;
  its output is console-only (summary + per-host log tails on
  failure).
- No auto-install of Docker, NVIDIA drivers, or Isaac Lab OS
  dependencies — those are operator prerequisites (documented, not
  automated).
- No integration with T4.1 (Valhalla). Bootstrap produces no bundles
  and no aggregate.
- No remote git-based provisioning. Like T3.1, we rsync the local
  working tree — consistent with the "local commits only" constraint.
- No support for heterogeneous fleets (different OS / CUDA versions).
  If a host's environment diverges from the controller's, bootstrap
  will still try, but no OS-level reconciliation happens.

## 4. Overview of changes

| Area | Change | File |
|---|---|---|
| Bootstrap core | `bootstrap_valkyrie(host, working_tree, *, ssh, rsync, build_timeout_s) -> BootstrapResult` + `bootstrap_fleet(fleet, working_tree, ...)` parallel driver | new `tools/odin/asgard/bootstrap.py` |
| Provisioner tweak | `_container_start` gains a `timeout_s` parameter (default 300 s, bootstrap passes 1800 s) | modify `tools/odin/asgard/provisioner.py` |
| CLI | `odin-bootstrap --fleet fleet.yaml [--build-timeout 1800] [--parallel] [--verbose]` | new `tools/odin/asgard/bootstrap_cli.py` |
| Package exports | Re-exports for `BootstrapResult`, `bootstrap_valkyrie`, `bootstrap_fleet` | modify `tools/odin/asgard/__init__.py` |
| Tests | Fake SSH/rsync unit tests covering happy path + every failure mode | new `tools/odin/tests/test_asgard_bootstrap.py`, new `tools/odin/tests/test_asgard_bootstrap_cli.py` |
| Docs | "Bootstrapping a fresh fleet" section | modify `tools/odin/README.md` |

**Unchanged (confirmed):**
- T3.1's preflight contract (`preflight_valkyrie`).
- T3.1's `run_dispatch` entry and `DispatchOptions`.
- T4.1's aggregator, writer, CLI.
- `fleet.yaml` schema.

## 5. CLI design

```bash
PYTHONPATH=. ./isaaclab.sh -p tools/odin/asgard/bootstrap_cli.py \
    --fleet fleet.yaml \
    [--build-timeout 1800] \
    [--sequential] \
    [--verbose]
```

- `--fleet fleet.yaml` — required, same schema T3.1 dispatch uses.
- `--build-timeout SECONDS` — timeout for `./docker/container.py start`
  (default `1800` = 30 min). Passed through to `_container_start`.
- `--sequential` — opt out of parallel-per-host execution (default is
  one thread per host). Useful if multiple hosts share a network pipe
  and parallel rsync saturates.
- `--verbose` — print per-step progress as each host works (SSH
  check, wipe, rsync, docker start, verify). Default prints only the
  final per-host summary.

**Exit code**: `0` if every host returned `ok=True`; `1` if any host
returned `ok=False`. The CLI also prints a final summary line
`bootstrap complete: K/N hosts ok`.

**Non-interactive**: the command never prompts. SSH host-key
verification uses the same `StrictHostKeyChecking=accept-new` policy
as the rest of Odin (T3.1's `ShellSSHRunner`).

## 6. Module design

### 6.1 `tools/odin/asgard/bootstrap.py`

```python
@dataclass
class BootstrapResult:
    host: str
    ok: bool
    message: str = ""
    commit_sha: str = ""
    step_durations_s: dict[str, float] = field(default_factory=dict)
    # Keys (in order): "wipe", "rsync", "container_start", "container_verify"


def bootstrap_valkyrie(
    host: ValkyrieConfig,
    working_tree: Path,
    *,
    ssh: SSHRunner,
    rsync: RsyncRunner,
    build_timeout_s: int = 1800,
) -> BootstrapResult:
    """Bring a fresh Valkyrie to T3.1-preflight-ready state.

    Flow:
      1. ssh reach — single echo probe with 15 s timeout.
      2. docker daemon reach — ``docker ps`` probe.
      3. wipe — ``rm -rf {isaaclab_path}`` (always; bootstrap is destructive
         by design so re-runs are idempotent).
      4. rsync — push controller working tree to ``{isaaclab_path}``.
      5. container start — ``cd {isaaclab_path} && ./docker/container.py
         start`` with ``build_timeout_s``.
      6. container verify — ``docker inspect -f '{{.State.Status}}'
         {container_name}``; must equal ``"running"``.

    Steps 1-2 short-circuit on failure (ssh or docker daemon problems
    are operator-fixable; no point rsync-ing to a broken host). Steps
    3-6 each get a step-duration entry in the result.

    Args:
        host: Target Valkyrie.
        working_tree: Controller-side IsaacLab path to push from.
        ssh: SSH runner.
        rsync: Rsync runner.
        build_timeout_s: Timeout for ``container.py start`` (default 1800 s).

    Returns:
        :class:`BootstrapResult` with ``ok=True`` iff all six steps passed.
    """


def bootstrap_fleet(
    fleet: Fleet,
    working_tree: Path,
    *,
    ssh: SSHRunner,
    rsync: RsyncRunner,
    build_timeout_s: int = 1800,
    parallel: bool = True,
    verbose: bool = False,
) -> list[BootstrapResult]:
    """Bootstrap every host in ``fleet``, optionally in parallel.

    Uses ``concurrent.futures.ThreadPoolExecutor(max_workers=len(fleet.hosts))``
    when ``parallel=True`` (default). Verbose mode prints
    ``[<host>] <step> <ok|fail> (<duration> s)`` as steps complete.

    Returns one :class:`BootstrapResult` per host, in fleet order.
    """
```

### 6.2 `tools/odin/asgard/provisioner.py` tweak

```python
def _container_start(host: ValkyrieConfig, ssh: SSHRunner, *, timeout_s: int = 300) -> bool:
    r = ssh.run(
        host,
        f"cd {host.isaaclab_path} && ./docker/container.py start",
        timeout_s=timeout_s,
    )
    return r.exit_code == 0
```

Callers:
- `provision_valkyrie` (T3.1 warm path) — unchanged, uses default 300 s.
- `bootstrap_valkyrie` — passes `timeout_s=build_timeout_s` (default 1800 s).

This is the only provisioner change.

### 6.3 `tools/odin/asgard/bootstrap_cli.py`

Argparse-wrapper mirroring `tools/odin/asgard/cli.py`'s shape:
```python
def parse_args(argv: list[str]) -> argparse.Namespace: ...
def main(argv: list[str] | None = None) -> int: ...
```

The CLI loads `fleet.yaml` via `load_fleet`, constructs `ShellSSHRunner`
/ `ShellRsyncRunner`, calls `bootstrap_fleet`, prints the summary,
and returns 0 / 1.

### 6.4 `tools/odin/asgard/__init__.py`

Re-export the three new public symbols alongside the existing T3.1 ones:

```python
from tools.odin.asgard.bootstrap import (
    BootstrapResult,
    bootstrap_fleet,
    bootstrap_valkyrie,
)
```

## 7. Failure handling

Per-host failure propagates into `BootstrapResult.ok=False`:

| Step fails | `message` | Recovery |
|---|---|---|
| ssh reach | `"ssh unreachable: <stderr>"` | operator fixes SSH, re-run bootstrap |
| docker daemon | `"docker daemon not responding: <stderr>"` | operator starts dockerd |
| wipe | `"failed to wipe {path}: <stderr>"` | usually permissions; operator checks |
| rsync | `"rsync push failed: <stderr>"` | typically disk space or network; operator checks |
| container_start | `"container.py start timed out after {n}s"` or stderr | extend `--build-timeout` or check docker build log remotely |
| container_verify | `"container {name!r} not running after start (status={s!r})"` | usually the build succeeded but startup failed; inspect container logs |

A failure on host A does NOT abort host B — they run independently
under the thread pool. The CLI's exit code is non-zero if any host
failed, so operators can `for host in $(bootstrap); do ...` safely.

Bootstrap does **not** write `dispatch.json` or any state file. All
output is console-only. Rationale: bootstrap is operator-driven
one-shot; state lives in the next T3.1 dispatch's `dispatch.json`.

## 8. Testing strategy

### 8.1 Unit tests `tools/odin/tests/test_asgard_bootstrap.py`

- `test_bootstrap_valkyrie_happy_path`: fake SSH + fake rsync all
  succeed; assert `ok=True`, all 6 step-durations populated, no
  error message.
- `test_bootstrap_valkyrie_ssh_unreachable`: fake SSH returns
  `exit_code != 0` on the first echo probe; assert `ok=False`,
  message contains "ssh unreachable", downstream steps not called
  (verify by asserting zero `rsync.push` calls).
- `test_bootstrap_valkyrie_docker_daemon_down`: fake SSH passes ssh
  reach but returns non-zero on `docker ps`; assert ok=False,
  downstream not called.
- `test_bootstrap_valkyrie_rsync_failure`: fake rsync returns
  non-zero; assert wipe ran before rsync; assert ok=False;
  container_start not called.
- `test_bootstrap_valkyrie_container_start_timeout`: fake SSH returns
  non-zero on `container.py start`; assert ok=False, message
  mentions timeout or failure, container_verify not called.
- `test_bootstrap_valkyrie_container_not_running_after_start`: fake
  SSH returns 0 on container.py start, but `docker inspect` returns
  `"exited"`. Assert ok=False with `"not running"` in message.
- `test_bootstrap_valkyrie_build_timeout_passed_through`: verify the
  `build_timeout_s=1800` value reaches the `_container_start` call
  (via fake SSH recording `timeout_s` kwargs).
- `test_bootstrap_valkyrie_step_durations_populated`: assert all four
  post-probe steps (wipe, rsync, container_start, container_verify)
  have non-zero durations recorded in the result.

### 8.2 Fleet-level tests

- `test_bootstrap_fleet_parallel`: 3 fake hosts; assert they run
  concurrently (by making one fake SSH block 100 ms and asserting total
  wall time ≈ max, not sum).
- `test_bootstrap_fleet_sequential`: `parallel=False`; same 3 hosts;
  assert total wall time ≈ sum (within a small tolerance).
- `test_bootstrap_fleet_mixed_outcome`: host A succeeds, host B's SSH
  fails. Assert one returns `ok=True`, the other `ok=False`; both
  appear in the returned list in fleet order.

### 8.3 CLI tests `tools/odin/tests/test_asgard_bootstrap_cli.py`

Mirror-parser style (same as T4.1 CLI tests):

- `test_parse_args_minimal`: only `--fleet`; defaults for other flags.
- `test_parse_args_all_flags`: every flag set.
- `test_main_exit_code_zero_when_all_ok`: monkeypatch
  `bootstrap_fleet` to return all-ok; assert exit code 0.
- `test_main_exit_code_one_when_any_fails`: monkeypatch to return one
  failure; assert exit code 1.
- `test_main_summary_line_printed`: capsys; assert `"bootstrap
  complete"` in stdout.

### 8.4 Regression guarantee

For each test that verifies a failure path, temporarily break the
corresponding production check and confirm the test fails — per the
`AGENTS.md` "regression tests must fail without the fix" rule.

## 9. How this fits with T3.1

**Operator workflow** on a fresh fleet:

```bash
# Step 1 — one-time (or "fresh machine") bootstrap.
PYTHONPATH=. ./isaaclab.sh -p tools/odin/asgard/bootstrap_cli.py --fleet fleet.yaml

# Step 2 — normal T3.1 dispatch (unchanged).
PYTHONPATH=. ./isaaclab.sh -p tools/odin/asgard/cli.py \
    --fleet fleet.yaml --physx-yaml ... --newton-yaml ... --seeds 42,43,44 ...
```

After bootstrap succeeds, every host satisfies T3.1's preflight
(`ssh_reach` + `docker_running` + `container_up` + `isaaclab_present`
all True). The dispatcher runs normally, including its own warm-path
`provision_valkyrie` which will rsync incrementally (no rebuild) and
skip container start (already `running`).

**When to re-bootstrap:** after a cold-boot of a Valkyrie, after a
long idle (image cache may be evicted), or when diagnostics point at
container drift. Also acceptable as a belt-and-braces "make sure
everything is fresh" periodic maintenance. Since bootstrap is
destructive (`rm -rf {isaaclab_path}` then re-rsync), it's not a
shortcut to "take a mid-experiment snapshot" — stop the dispatch
first.

## 10. Real-fleet validation tie-in

The T4.1 spec (§10) had a "real-fleet validation" step blocked on
T3.1 being able to handle the two fresh Valkyries. This spec unblocks
that. Concrete next steps after bootstrap lands:

1. `odin-bootstrap --fleet fleet.yaml` against 10.176.221.98 +
   10.63.172.46 — ~20-30 min.
2. Run a single-task dry dispatch (1 task × 1 seed × 1 backend) to
   confirm end-to-end.
3. Run the full T4.1 validation dispatch (5 tasks × 3 seeds × 2
   backends).
4. Inspect the resulting `aggregate.json`.

## 11. Out of scope

- No auto-install of Docker / NVIDIA drivers / OS packages.
- No health-monitoring daemon or heartbeat pinger on Valkyries.
- No incremental re-bootstrap (diff the tree, only re-rsync changed
  files) — rsync itself already does incremental for non-fresh cases;
  bootstrap's wipe+re-rsync is deliberate.
- No Docker image pre-pull or layer-cache priming — `container.py
  start` handles it.
- No `odin-bootstrap verify` sub-command (use T3.1 preflight for that;
  running a no-job `odin-dispatch` will preflight then exit cleanly).
- No GPU smoke test (e.g. `nvidia-smi` inside the container) — T3.1's
  subsequent preflight + the first actual dispatch surface this.
- No cross-platform support (assume Linux on both controller and
  Valkyries).
