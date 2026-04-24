# Odin T3.1 Integration Fixes — Design

**Status:** approved
**Date:** 2026-04-24
**Task covered:** Three coupled fixes that surfaced on the first real-fleet
dispatch attempt. T3.1 was shipped with fake-SSH/Rsync unit tests; the
fakes elided the Hugin ↔ worker wire format, so three architectural gaps
only appear on real hardware. Each is small, but they must land together
— any one alone leaves the dispatch broken.

## 1. Motivation

On 2026-04-24 the first real `odin-dispatch` against a two-host fleet
(10.176.221.98, 10.63.172.46) failed all four smoke jobs with a mix of
`hugin_crash` and `infrastructure` errors. Root-cause analysis across
the failures points at three architectural gaps in T3.1:

### Gap 1 — Hugin/Munin generate their own `run_id`

`tools/odin/hugin/run.py:83` and the analogous line in `munin/run.py`
compute `run_id = compute_run_id(framework, backend, task, seed, now=run_start)`
where `run_start = datetime.now()`. T3.1's dispatcher also computes a
`run_id` (via `tools/odin/asgard/jobs._make_run_id`, using
`dispatch_id`). The two disagree by the time-delta between dispatch
setup and Hugin start — typically 30 s-2 min — producing two
different `<YYYYMMDD-HHMMSS>` stems in the run_id string.

The dispatcher records its version in `dispatch.json` and expects to
rsync back `~/IsaacLab/odin_runs/<dispatcher_run_id>/`. Hugin writes
to `odin_runs/<hugin_run_id>/`. The two paths never collide.

### Gap 2 — Bundles land inside a container-only path

Hugin is invoked as `docker exec <container> bash -lc 'cd
/workspace/isaaclab && ./isaaclab.sh -p tools/odin/hugin/run.py ...
--runs_root odin_runs'`. The relative `odin_runs` resolves under
`/workspace/isaaclab/odin_runs/`, which is *not* bind-mounted. The
bundle is written to the container's overlay filesystem and is
invisible to the host.

T3.1's worker runs `rsync horde@host:~/IsaacLab/odin_runs/<run_id>/
<dispatch_dir>/<run_id>/` to pull the bundle back — but
`~/IsaacLab/odin_runs/` doesn't exist on the host and never did,
because nothing writes there.

### Gap 3 — Worker doesn't propagate `run_id` to Hugin

Worker's `_build_docker_exec_cmd` passes every other Hugin/Munin arg
(`--task`, `--backend`, `--seed`, `--num_envs`, `--max_iterations`,
`--runs_root`) but not `--run_id`. Even if Hugin accepted `--run_id`,
the worker wouldn't supply it. Hugin's CLI today has no such flag.

The three gaps compound: the worker computes its run_id, records it,
and expects to find its bundle. Hugin (inside the container) ignores
the worker's run_id, writes to a different path, and the worker's
rsync-pull succeeds 0% of the time.

## 2. Goals

- Hugin and Munin accept a `--run_id <string>` CLI argument. When
  provided, it is used verbatim as the bundle directory name; when
  absent, current behavior is preserved (compute from timestamp).
- T3.1's worker passes `--run_id <job.run_id>` on every Hugin/Munin
  invocation so bundle path and dispatcher expectation match.
- Bundles written inside the container land in a path that's visible
  on the host — added as a new bind-mount in
  `docker/docker-compose.yaml` for the `isaac-lab-base` profile.
- Bootstrap pre-creates the host-side bundle dir so the bind-mount
  target exists before `container.py start`.
- Real-fleet dispatch smoke (4 jobs across 2 hosts, 1 task × 2 seeds ×
  2 backends) completes end-to-end with zero `infrastructure` /
  `hugin_crash` failures.

## 3. Non-goals

- No redesign of `run_id` format. Hugin's `compute_run_id` stays as
  the fallback; we're just adding an override.
- No changes to the Asgard state machine, resume semantics, failure
  classification, or preflight logic.
- No changes to Hugin/Munin's bundle content (manifest / training /
  startup JSONs, training_data/ layout) — only the location where the
  bundle root lands.
- No schema bump on `dispatch.json` or the Odin bundle schema.
- No support for a *different* bundle root on each host (all
  Valkyries use the same container mount path).
- No tests for the Dockerfile change — `docker-compose.yaml` edits
  are validated by the real-fleet smoke pass, not by unit tests.

## 4. Overview of changes

| Area | Change | File(s) |
|---|---|---|
| Gap 1 — Hugin/Munin CLI | Add `--run_id` to both Hugin and Munin wrappers; use override or fall back to `compute_run_id`. (The downstream benchmark scripts already accept `--run_id`; Hugin/Munin already forward it. The gap is purely at the wrapper CLI layer.) | `tools/odin/hugin/run.py`, `tools/odin/munin/run.py` |
| Gap 2 — bind-mount | Add `- ${PWD}/odin_runs:/workspace/isaaclab/odin_runs` to `isaac-lab-base` service volumes | `docker/docker-compose.yaml` |
| Gap 2 (cont.) — pre-create | `mkdir -p {isaaclab_path}/odin_runs` on remote in bootstrap, BEFORE `container.py start` so docker-compose finds the mount source | `tools/odin/asgard/bootstrap.py` |
| Gap 3 — worker propagates run_id | `_build_docker_exec_cmd` appends `--run_id <job.run_id>` | `tools/odin/asgard/worker.py` |
| Tests | Hugin/Munin CLI override unit tests; worker command-shape update | `tools/odin/tests/test_hugin.py`, `tools/odin/tests/test_munin.py`, `tools/odin/tests/test_asgard_worker.py` |

## 5. Hugin / Munin `--run_id` CLI

### 5.1 Hugin (`tools/odin/hugin/run.py`)

Add to the existing argparse block:

```python
parser.add_argument(
    "--run_id",
    type=str,
    default=None,
    help=(
        "Override the computed run_id for this bundle. When set, Hugin "
        "uses this string verbatim as the bundle directory name. "
        "Intended for Odin's T3.1 dispatcher which pre-computes run_ids "
        "against its dispatch_id so all bundles under a dispatch share "
        "a consistent timestamp stem. When unset, Hugin falls back to "
        "``compute_run_id(framework, backend, task, seed, now)``."
    ),
)
```

Update the run_id line (~83):

```python
run_id = args.run_id or compute_run_id(
    "rsl_rl", args.backend, args.task, args.seed, now=run_start,
)
```

### 5.2 Munin (`tools/odin/munin/run.py`)

Same change with `"skrl"` in place of `"rsl_rl"`.

### 5.3 Tests

Append to `tools/odin/tests/test_hugin.py`:

```python
def test_hugin_honors_run_id_override(tmp_path, monkeypatch):
    """--run_id uses the string verbatim instead of compute_run_id."""
    bundle_root = str(tmp_path)
    monkeypatch.setattr(hugin_run, "_subprocess_run", _fake_run_factory())
    monkeypatch.setattr(
        "sys.argv",
        [
            "hugin",
            "--task", "Isaac-Ant-Direct-v0",
            "--backend", "physx",
            "--seed", "42",
            "--runs_root", bundle_root,
            "--skip_startup",
            "--run_id", "dispatched-run-id-xyz",
        ],
    )
    hugin_run.main()
    assert os.path.isdir(os.path.join(bundle_root, "dispatched-run-id-xyz"))
    # The auto-generated run_id prefix should NOT appear.
    other = [d for d in os.listdir(bundle_root) if d != "dispatched-run-id-xyz"]
    assert other == []
```

Mirror test for Munin.

## 6. docker-compose bind-mount

### 6.1 `docker/docker-compose.yaml`

Find the `isaac-lab-base` service's `volumes` section. Add:

```yaml
      - type: bind
        source: ../odin_runs
        target: /workspace/isaaclab/odin_runs
        bind:
          create_host_path: true
```

The `create_host_path: true` is a belt-and-braces: docker-compose will
create `../odin_runs` on the host if it doesn't exist, so the mount
works on a fresh Valkyrie. The bootstrap also explicitly creates the
dir (§7 below) — both paths are OK; layered defenses are fine.

Path note: `source: ../odin_runs` is relative to `docker/` (where
compose is invoked), so it resolves to the IsaacLab repo root's
`odin_runs/`. On a Valkyrie, that's `/home/horde/IsaacLab/odin_runs/`.

### 6.2 Why docker-compose, not Odin-only overrides

docker-compose supports per-file `-f override.yaml` layering; we
*could* ship an `odin.override.yaml` alongside `docker-compose.yaml`
and make Odin's `container.py start` calls pass `-f` to layer it on.
That avoids modifying the shared `docker-compose.yaml`.

Rejected because:

- `./docker/container.py start` doesn't accept `-f overrides` as a CLI
  knob, and wrapping it would bring this into Odin's surface area.
- `odin_runs/` is a stable, small, well-named output directory. The
  top-level `docker-compose.yaml` already bind-mounts `logs/` and
  `data_storage/`; adding `odin_runs/` follows the same pattern.

This change is a one-line addition to a config file; it cannot break
non-Odin users (the directory is created on-demand by
`create_host_path: true`).

## 7. Bootstrap pre-creates `odin_runs/`

In `tools/odin/asgard/bootstrap.py`, between step 4b (`configure_headless`)
and step 5 (`container_start`), add:

```python
# 4c. Pre-create the bundle output directory on the host. docker-compose's
# isaac-lab-base service bind-mounts ~/IsaacLab/odin_runs →
# /workspace/isaaclab/odin_runs so Hugin/Munin bundles written inside the
# container land on the host and can be rsync-pulled by the dispatcher.
# `create_host_path: true` in the compose file would also auto-create it,
# but doing it explicitly here means the directory is owned by ``horde``
# (not root, as it would be when docker creates it).
t0 = _time_step()
r = ssh.run(
    host,
    f"mkdir -p {host.isaaclab_path}/odin_runs",
    timeout_s=15,
)
step_durations_s["create_odin_runs"] = _time_step() - t0
if r.exit_code != 0:
    return BootstrapResult(
        host=host.host,
        ok=False,
        message=f"failed to create {host.isaaclab_path}/odin_runs: {r.stderr.strip() or 'non-zero exit'}",
        commit_sha=commit_sha,
        step_durations_s=step_durations_s,
    )
```

Update the happy-path test's expected `step_durations_s` keys to
include `create_odin_runs`.

## 8. Worker propagates `run_id`

In `tools/odin/asgard/worker.py`, the `_build_docker_exec_cmd` method
(or equivalent, depending on current shape) builds the docker-exec
command. Add `--run_id <job.run_id>` to the Hugin/Munin argv.

Concretely, after the existing `--runs_root odin_runs` entry:

```python
runner_args = [
    "--task", job.task_id,
    "--backend", job.backend,
    "--seed", str(job.seed),
    "--num_envs", str(job.num_envs),
    "--max_iterations", str(job.max_iterations),
    "--runs_root", "odin_runs",
    "--run_id", job.run_id,   # NEW
]
```

`job.run_id` is the dispatcher's pre-computed run_id (the one in
`dispatch.json` and the one the worker's rsync-pull will look for).
Hugin uses it verbatim, writes the bundle at
`/workspace/isaaclab/odin_runs/<job.run_id>/`, which maps via the
bind-mount to `~/IsaacLab/odin_runs/<job.run_id>/` on the host — where
the worker's rsync-pull finds it.

### 8.1 Test

Update `tools/odin/tests/test_asgard_worker.py` to assert `--run_id`
appears in the constructed command with the expected value.

## 9. Rsync-pull path (already correct after §6)

With `§6`'s bind-mount and `§8`'s run_id propagation, the bundle lands
at `~/IsaacLab/odin_runs/<job.run_id>/` on the host. The worker's
existing rsync-pull string is
`horde@host:~/IsaacLab/odin_runs/<run_id>/ → <dispatch_dir>/<run_id>/`.
That remains unchanged; it just starts finding bundles where it expects.

## 10. Testing strategy

### 10.1 Unit tests (existing green bar + new)

- `test_hugin_honors_run_id_override` (new).
- `test_munin_honors_run_id_override` (new).
- `test_build_docker_exec_cmd_includes_run_id` (new in
  `test_asgard_worker.py`, or update an existing test).
- `test_bootstrap_valkyrie_happy_path`'s `step_durations_s` assertion
  extended with `create_odin_runs`.
- `test_bootstrap_valkyrie_creates_odin_runs_dir` (new).

### 10.2 Real-fleet validation

Re-run the smoke dispatch (4 jobs, 1 task × 2 seeds × 2 backends)
against the two Valkyries:

```bash
PYTHONPATH=. ./isaaclab.sh -p tools/odin/asgard/cli.py \
    --fleet fleet.yaml \
    --physx-yaml tools/odin/config/physx_envs.yaml \
    --newton-yaml tools/odin/config/newton_envs.yaml \
    --seeds 42,43 --include 'Isaac-Ant-Direct-v0' --verbose
```

**Success criteria** (all must hold):

- 4/4 jobs report `status=completed` in `dispatch.json`.
- Each bundle dir on the controller contains `manifest.json`,
  `training.json`, `startup.json`, and `training_data/`.
- `odin_runs/<dispatch_id>/aggregate.json` lands with 1 row, 4 seeds
  split physx/newton, no failures.
- No `infrastructure` or `hugin_crash` entries in the dispatch
  summary.

### 10.3 Re-bootstrap needed?

Yes — because (a) the existing containers were started with an image
that doesn't have the new bind-mount, and (b) the host `odin_runs/`
dir doesn't exist. `odin-bootstrap --fleet fleet.yaml` will:

1. Wipe + re-rsync the working tree (picks up the bootstrap's new
   `create_odin_runs` step and the updated `docker-compose.yaml`).
2. Run `./docker/container.py start` — re-creates the container with
   the new bind-mount. No rebuild necessary; compose starts a new
   container using the existing image, with updated volume config.

## 11. Out of scope

- No override mechanism to name the bind-mount target per-host.
- No cleanup of stale bundles on Valkyries. (Bootstrap's `--fresh`
  wipes everything including bundles; otherwise they accumulate.)
- No `/isaac-sim` fallback if the container's bind-mount target is
  missing — docker-compose's `create_host_path: true` handles that
  case defensively.
- No changes to T4.1's aggregator. It already reads bundles from
  `<dispatch_dir>/<run_id>/` on the controller; the fix above makes
  sure they actually land there.
- No deprecation of `compute_run_id`. It stays as the fallback when
  `--run_id` is unset (standalone Hugin runs without a dispatcher).
