# Odin T3.1 Integration Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `odin-dispatch` actually produce usable bundles on a real fleet by closing three coupled T3.1 gaps: Hugin/Munin don't accept `--run_id`, bundles are written to a container-only path, and the worker doesn't propagate the dispatcher's run_id.

**Architecture:** Four small code changes + one docker-compose edit + a real-fleet smoke re-run. (1) Hugin/Munin get `--run_id` override that falls back to `compute_run_id` when unset. (2) `docker/docker-compose.yaml` bind-mounts `../odin_runs` into the container. (3) Bootstrap pre-creates `~/IsaacLab/odin_runs/` on each host so the bind-mount source exists and is owned by the SSH user. (4) `asgard/worker.py:_build_docker_exec_cmd` appends `--run_id <job.run_id>` so Hugin uses the dispatcher's id. Together: bundles land on the host at the path the dispatcher's rsync-pull expects.

**Tech Stack:** Python 3.10+ (stdlib argparse), pytest, Docker Compose v2.

**Spec:** `docs/superpowers/specs/2026-04-24-odin-t3-1-integration-fixes-design.md`

---

## File Structure

**Modified:**
- `tools/odin/hugin/run.py` — new `--run_id` argparse flag; use it when set, else `compute_run_id(...)`.
- `tools/odin/munin/run.py` — same pattern as Hugin.
- `tools/odin/tests/test_hugin.py` — new `test_hugin_honors_run_id_override`.
- `tools/odin/tests/test_munin.py` — new `test_munin_honors_run_id_override`.
- `docker/docker-compose.yaml` — one new bind-mount entry in `x-default-isaac-lab-volumes`.
- `tools/odin/asgard/bootstrap.py` — new step 4c `create_odin_runs` between `configure_headless` and `container_start`.
- `tools/odin/tests/test_asgard_bootstrap.py` — update happy-path's `step_durations_s` keys; add `test_bootstrap_valkyrie_creates_odin_runs_dir`.
- `tools/odin/asgard/worker.py` — add `--run_id {job.run_id}` to `_build_docker_exec_cmd`.
- `tools/odin/tests/test_asgard_worker.py` — update existing command-shape assertion to expect `--run_id`.

**Unchanged (confirmed):**
- `scripts/benchmarks/benchmark_rsl_rl.py` + `benchmark_skrl.py` — already accept `--run_id`; Hugin/Munin already forward it via `training_cmd`. The gap is purely at the wrapper CLI surface.
- `tools/odin/asgard/jobs.py` — `_make_run_id` already composes `{framework_slug}_{backend}_{task_id}_{dispatch_id}_seed{seed}`; that is exactly what we want to pass.
- `tools/odin/asgard/runner.py` — rsync-pull path (`~/IsaacLab/odin_runs/<run_id>/`) is already correct; it just starts finding bundles once §6+§8 land.

**Task ordering rationale:** Task 1 is the wrapper CLI change (Hugin + Munin together — they're parallel). Task 2 is the docker-compose edit (small, independent). Task 3 is the bootstrap pre-create step. Task 4 is the worker update (depends on Task 1's `--run_id` existing). Task 5 is the real-fleet smoke pass + re-bootstrap.

---

### Task 1: Hugin + Munin `--run_id` override

**Files:**
- Modify: `tools/odin/hugin/run.py` (argparse block + run_id computation line)
- Modify: `tools/odin/munin/run.py` (same shape)
- Modify: `tools/odin/tests/test_hugin.py` (append one test)
- Modify: `tools/odin/tests/test_munin.py` (append one test)

- [ ] **Step 1: Read the current Hugin argparse section**

```bash
grep -n "parser.add_argument\|run_id = compute" tools/odin/hugin/run.py | head
```

Expected: a sequence of `parser.add_argument(...)` calls around lines 78-92, and a `run_id = compute_run_id(...)` around line 83.

- [ ] **Step 2: Write the failing Hugin test**

Append to `tools/odin/tests/test_hugin.py`:

```python
def test_hugin_honors_run_id_override(tmp_path, monkeypatch):
    """--run_id uses the string verbatim instead of compute_run_id."""
    bundle_root = str(tmp_path)
    fake_run = _fake_run_factory()
    monkeypatch.setattr(hugin_run, "_subprocess_run", fake_run)
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
    # No auto-generated run_id sibling directory.
    siblings = [d for d in os.listdir(bundle_root) if d != "dispatched-run-id-xyz"]
    assert siblings == [], f"unexpected sibling bundle dirs: {siblings}"
```

- [ ] **Step 3: Run the Hugin test to verify it fails**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_hugin.py::test_hugin_honors_run_id_override -v --confcutdir=tools/odin
```

Expected: FAIL — either `--run_id` is an unrecognised arg, or the run_id override isn't applied and the bundle lands under a `rsl-rl_physx_..._<auto-timestamp>_seed42` dir (triggering the `siblings == []` assertion).

- [ ] **Step 4: Add `--run_id` argparse flag + use it in `run.py`**

In `tools/odin/hugin/run.py`, find the existing argparse block (around lines 78-92). Add a new argument after the existing `--task / --backend / --seed / --num_envs / --max_iterations / --runs_root / --ema_alpha / --no_series / --skip_startup` arguments:

```python
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help=(
            "Override the computed run_id for this bundle. When set, Hugin "
            "uses this string verbatim as the bundle directory name. "
            "Intended for Odin's T3.1 dispatcher, which pre-computes "
            "run_ids against its dispatch_id so all bundles under one "
            "dispatch share a consistent timestamp stem. When unset, "
            "Hugin falls back to ``compute_run_id(framework, backend, "
            "task, seed, now)``."
        ),
    )
```

Then replace the run_id line (~83):

```python
    run_id = compute_run_id("rsl_rl", args.backend, args.task, args.seed, now=run_start)
```

with:

```python
    run_id = args.run_id or compute_run_id(
        "rsl_rl", args.backend, args.task, args.seed, now=run_start,
    )
```

- [ ] **Step 5: Run the Hugin test to verify it passes**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_hugin.py -v --confcutdir=tools/odin
```

Expected: all hugin tests pass (existing + 1 new).

- [ ] **Step 6: Mirror the change in Munin**

In `tools/odin/munin/run.py`, add the same `--run_id` argparse block right alongside the existing flags (after `--skip_startup`):

```python
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help=(
            "Override the computed run_id for this bundle. When set, Munin "
            "uses this string verbatim as the bundle directory name. "
            "Intended for Odin's T3.1 dispatcher, which pre-computes "
            "run_ids against its dispatch_id so all bundles under one "
            "dispatch share a consistent timestamp stem. When unset, "
            "Munin falls back to ``compute_run_id(framework, backend, "
            "task, seed, now)``."
        ),
    )
```

Find the existing `run_id = compute_run_id(...)` line (Munin uses `"skrl"` as the framework):

```python
    run_id = compute_run_id("skrl", args.backend, args.task, args.seed, now=run_start)
```

Replace with:

```python
    run_id = args.run_id or compute_run_id(
        "skrl", args.backend, args.task, args.seed, now=run_start,
    )
```

Append to `tools/odin/tests/test_munin.py`:

```python
def test_munin_honors_run_id_override(tmp_path, monkeypatch):
    """--run_id uses the string verbatim instead of compute_run_id."""
    bundle_root = str(tmp_path)
    fake_run = _fake_run_factory()
    monkeypatch.setattr(munin_run, "_subprocess_run", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "munin",
            "--task", "Isaac-Ant-Direct-v0",
            "--backend", "physx",
            "--seed", "42",
            "--runs_root", bundle_root,
            "--skip_startup",
            "--run_id", "dispatched-run-id-xyz",
        ],
    )
    munin_run.main()

    assert os.path.isdir(os.path.join(bundle_root, "dispatched-run-id-xyz"))
    siblings = [d for d in os.listdir(bundle_root) if d != "dispatched-run-id-xyz"]
    assert siblings == [], f"unexpected sibling bundle dirs: {siblings}"
```

- [ ] **Step 7: Run the full Hugin + Munin test suites**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_hugin.py tools/odin/tests/test_munin.py -v --confcutdir=tools/odin
```

Expected: all 4 tests pass (2 existing + 2 new per file).

- [ ] **Step 8: Pre-commit + commit**

```bash
./isaaclab.sh -f
# Re-run until clean if hooks auto-fix.
git add tools/odin/hugin/run.py tools/odin/munin/run.py tools/odin/tests/test_hugin.py tools/odin/tests/test_munin.py
git commit -m "Hugin/Munin: accept --run_id CLI override"
```

Subject is 42 chars.

---

### Task 2: docker-compose bind-mount for `odin_runs/`

**Files:**
- Modify: `docker/docker-compose.yaml` (add one bind-mount entry to the shared `x-default-isaac-lab-volumes` anchor)

- [ ] **Step 1: Locate the insertion point**

```bash
grep -n "x-default-isaac-lab-volumes\|.bash_history\|x-default-isaac-lab-environment" docker/docker-compose.yaml | head
```

Expected: the `x-default-isaac-lab-volumes` anchor opens around line 10; the last bind-mount in the anchor is for `.bash_history` at around line 69-71, and `x-default-isaac-lab-environment` starts after.

- [ ] **Step 2: Read the exact block around `.bash_history` to pick the precise insertion anchor**

Run `sed -n '65,75p' docker/docker-compose.yaml`. The block looks like:

```yaml
    # This volume is used to store the history of the bash shell
  - type: bind
    source: .isaac-lab-docker-history
    target: ${DOCKER_USER_HOME}/.bash_history

x-default-isaac-lab-environment: &default-isaac-lab-environment
```

- [ ] **Step 3: Insert the `odin_runs` bind-mount immediately after the `.bash_history` entry, before the blank line that precedes `x-default-isaac-lab-environment`**

Change

```yaml
    # This volume is used to store the history of the bash shell
  - type: bind
    source: .isaac-lab-docker-history
    target: ${DOCKER_USER_HOME}/.bash_history

x-default-isaac-lab-environment: &default-isaac-lab-environment
```

to

```yaml
    # This volume is used to store the history of the bash shell
  - type: bind
    source: .isaac-lab-docker-history
    target: ${DOCKER_USER_HOME}/.bash_history
    # Bundle output dir for Odin's dispatcher. Hugin/Munin write bundles
    # into this path inside the container; bind-mounting makes them
    # visible on the host so T3.1's worker can rsync them back.
  - type: bind
    source: ../odin_runs
    target: ${DOCKER_ISAACLAB_PATH}/odin_runs
    bind:
      create_host_path: true

x-default-isaac-lab-environment: &default-isaac-lab-environment
```

- [ ] **Step 4: Sanity-check the YAML parses**

```bash
./isaaclab.sh -p -c "import yaml; yaml.safe_load(open('docker/docker-compose.yaml')); print('yaml ok')"
```

Expected: prints `yaml ok`.

- [ ] **Step 5: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add docker/docker-compose.yaml
git commit -m "Docker: bind-mount odin_runs into isaac-lab-base"
```

Subject is 48 chars.

---

### Task 3: Bootstrap pre-creates `odin_runs/`

**Files:**
- Modify: `tools/odin/asgard/bootstrap.py` (add new step between `configure_headless` and `container_start`)
- Modify: `tools/odin/tests/test_asgard_bootstrap.py` (update happy-path's `step_durations_s` keys; add one new test)

- [ ] **Step 1: Read the current `step_durations_s` assertion in the happy-path test**

```bash
grep -n "step_durations_s.keys\|configure_headless" tools/odin/tests/test_asgard_bootstrap.py | head -5
```

Expected: the assertion is at around line 114, asserting the set `{"wipe", "rsync", "configure_headless", "fix_isaac_sim_symlink", "container_start", "container_verify"}`.

- [ ] **Step 2: Update the happy-path assertion to include `create_odin_runs`**

Find the block:

```python
    assert set(result.step_durations_s.keys()) == {
        "wipe",
        "rsync",
        "configure_headless",
        "fix_isaac_sim_symlink",
        "container_start",
        "container_verify",
    }
```

Replace with:

```python
    assert set(result.step_durations_s.keys()) == {
        "wipe",
        "rsync",
        "configure_headless",
        "create_odin_runs",
        "container_start",
        "fix_isaac_sim_symlink",
        "container_verify",
    }
```

Note: `create_odin_runs` sits between `configure_headless` and `container_start` in pipeline order; `fix_isaac_sim_symlink` stays after `container_start` (it docker-execs into the running container). The order in the set literal is cosmetic — the assertion compares sets.

- [ ] **Step 3: Write the failing test for the new step**

Append to `tools/odin/tests/test_asgard_bootstrap.py`:

```python
def test_bootstrap_valkyrie_creates_odin_runs_dir(tmp_path: Path):
    """bootstrap must `mkdir -p {isaaclab_path}/odin_runs` after configure_headless."""
    ssh = _FakeSSH(
        replies={"docker inspect": 0},
        reply_stdout={"docker inspect": "running"},
    )
    rsync = _FakeRsync()
    bootstrap_valkyrie(_host(), tmp_path, ssh=ssh, rsync=rsync)
    mkdir_calls = [
        c for c in ssh.calls
        if "mkdir -p" in c.cmd and "/opt/IsaacLab/odin_runs" in c.cmd
    ]
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
```

- [ ] **Step 4: Run to verify the new tests fail**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_bootstrap.py::test_bootstrap_valkyrie_creates_odin_runs_dir tools/odin/tests/test_asgard_bootstrap.py::test_bootstrap_valkyrie_create_odin_runs_failure tools/odin/tests/test_asgard_bootstrap.py::test_bootstrap_valkyrie_happy_path -v --confcutdir=tools/odin
```

Expected: the two new tests FAIL (no mkdir_calls yet); the happy-path test FAILS on the `step_durations_s` assertion (missing `create_odin_runs` key).

- [ ] **Step 5: Add the step to `bootstrap.py`**

In `tools/odin/asgard/bootstrap.py`, find the `# 5. Container start.` line. Immediately before it (i.e. after the `configure_headless` block ends), insert:

```python
    # 4c. Pre-create the bundle output directory on the host.
    # docker-compose's isaac-lab-base service bind-mounts ``~/IsaacLab/odin_runs``
    # to ``/workspace/isaaclab/odin_runs``, so Hugin/Munin bundles written inside
    # the container land on the host and can be rsync-pulled by the dispatcher.
    # compose's ``create_host_path: true`` would auto-create it, but doing it
    # explicitly here means the directory is owned by ``{host.ssh_user}``
    # (not root, as docker would do it).
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

- [ ] **Step 6: Run all bootstrap tests to verify pass**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_bootstrap.py -v --confcutdir=tools/odin
```

Expected: all tests pass (17 pre-existing + 2 new = 19).

- [ ] **Step 7: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/bootstrap.py tools/odin/tests/test_asgard_bootstrap.py
git commit -m "Bootstrap: pre-create odin_runs/ on remote"
```

Subject is 41 chars.

---

### Task 4: Worker propagates `--run_id` to Hugin/Munin

**Files:**
- Modify: `tools/odin/asgard/worker.py` (add `--run_id` entry to `inner_parts` in `_build_docker_exec_cmd`)
- Modify: `tools/odin/tests/test_asgard_worker.py` (update or add a test for the new arg)

- [ ] **Step 1: Read the current command-shape test to find the right anchor**

```bash
grep -n "def test_build_docker_exec_cmd\|--runs_root\|run_id" tools/odin/tests/test_asgard_worker.py | head
```

Expected: at least one test exercising `_build_docker_exec_cmd`, with assertions about the command string containing args like `--task`, `--runs_root`, etc.

- [ ] **Step 2: Write the new assertion**

If there is an existing test `test_build_docker_exec_cmd_contains_expected_args` (or similar name) exercising `_build_docker_exec_cmd`, **extend** its assertions to also require `--run_id` appears with the job's run_id:

```python
    assert f"--run_id {job.run_id}" in cmd
```

If no such assertion exists, append a dedicated test at the bottom of `tools/odin/tests/test_asgard_worker.py`:

```python
def test_build_docker_exec_cmd_includes_run_id():
    """worker must pass --run_id so Hugin/Munin write bundles at the dispatcher-expected path."""
    from tools.odin.asgard.fleet import ValkyrieConfig
    from tools.odin.asgard.jobs import JobEntry
    from tools.odin.asgard.worker import _build_docker_exec_cmd

    host = ValkyrieConfig(
        host="v1.internal",
        ssh_user="odin",
        ssh_key=None,
        isaaclab_path="/home/odin/IsaacLab",
    )
    job = JobEntry(
        run_id="rsl-rl_physx_Isaac-Ant-Direct-v0_20260424-120000_seed42",
        task_id="Isaac-Ant-Direct-v0",
        framework="rsl_rl",
        backend="physx",
        num_envs=4096,
        max_iterations=300,
        seed=42,
        bundle_dir_name="rsl-rl_physx_Isaac-Ant-Direct-v0_20260424-120000_seed42",
    )
    cmd = _build_docker_exec_cmd(host, job)
    assert f"--run_id {job.run_id}" in cmd
    # Sanity: other expected args still present.
    assert f"--task {job.task_id}" in cmd
    assert f"--backend {job.backend}" in cmd
    assert "--runs_root odin_runs" in cmd
```

- [ ] **Step 3: Run the test to verify it fails**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_worker.py -v --confcutdir=tools/odin
```

Expected: the new or extended test FAILS on the `--run_id` assertion because `_build_docker_exec_cmd` doesn't yet include that arg.

- [ ] **Step 4: Edit `_build_docker_exec_cmd` to pass `--run_id`**

In `tools/odin/asgard/worker.py`, find the `inner_parts` list construction around line 60. The current last entry is `"--runs_root odin_runs"`. Add a new entry right after it:

```python
    inner_parts = [
        "cd /workspace/isaaclab",
        "PYTHONPATH=.",
        f"./isaaclab.sh -p {runner_script}",
        f"--task {job.task_id}",
        f"--backend {job.backend}",
        f"--seed {job.seed}",
        f"--num_envs {job.num_envs}",
        f"--max_iterations {job.max_iterations}",
        "--runs_root odin_runs",
        f"--run_id {job.run_id}",
    ]
```

- [ ] **Step 5: Run all worker tests to confirm pass**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/test_asgard_worker.py -v --confcutdir=tools/odin
```

Expected: all pass (existing + 1 new / 1 updated).

- [ ] **Step 6: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add tools/odin/asgard/worker.py tools/odin/tests/test_asgard_worker.py
git commit -m "Worker: pass --run_id to Hugin/Munin docker-exec"
```

Subject is 50 chars (at the cap).

---

### Task 5: Full-suite sweep + real-fleet smoke

**Files:** none modified.

- [ ] **Step 1: Run the entire Odin + benchmarks test suite**

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/ scripts/benchmarks/tests/ --confcutdir=tools/odin 2>/dev/null | tail -5
```

Expected: all pass, ~200+ tests. If any pre-existing test that exercised `_build_docker_exec_cmd` now fails because its assertion didn't know about `--run_id`, extend it.

- [ ] **Step 2: Pre-commit**

```bash
./isaaclab.sh -f
```

Expected: clean.

- [ ] **Step 3: Re-bootstrap the two Valkyries**

Existing containers were started *before* this plan's docker-compose bind-mount change. Docker-compose reads the compose file every time `up` (or `container.py start`) runs; it restarts the container if its config (including mounts) has changed. The rsync from re-bootstrap also pushes the updated `docker-compose.yaml` to the remote. Steps:

```bash
PYTHONPATH=. ./isaaclab.sh -p tools/odin/asgard/bootstrap_cli.py --fleet fleet.yaml --verbose --build-timeout 2700
```

Expected: `bootstrap complete: 2/2 hosts ok`. Because the isaac-lab-base image already exists on both hosts, this skips the slow docker build and just restarts the container with the new volume config — typically 2-3 min per host.

- [ ] **Step 4: Sanity-check the bind-mount is live on one host**

```bash
ssh -i ~/.ssh/id_ed25519 horde@10.176.221.98 'docker inspect isaac-lab-base --format "{{range .Mounts}}{{.Type}} {{.Source}} -> {{.Destination}}{{println}}{{end}}"' | grep odin_runs
```

Expected: one line like `bind /home/horde/IsaacLab/odin_runs -> /workspace/isaaclab/odin_runs`.

- [ ] **Step 5: Run the smoke dispatch**

```bash
PYTHONPATH=. ./isaaclab.sh -p tools/odin/asgard/cli.py \
    --fleet fleet.yaml \
    --physx-yaml tools/odin/config/physx_envs.yaml \
    --newton-yaml tools/odin/config/newton_envs.yaml \
    --seeds 42,43 \
    --include 'Isaac-Ant-Direct-v0' \
    --verbose \
    --per-job-timeout 1800
```

Expected: all 4 jobs complete; final summary shows `4 completed, 0 failed`. Each `odin_runs/<dispatch_id>/<run_id>/` bundle dir contains `manifest.json`, `training.json`, `startup.json`, `training_data/`. `aggregate.json` lands at the dispatch dir root with 1 row and 4 seeds (2 physx + 2 newton) all completed.

If any job fails: the failure message should be a real training-layer problem (e.g. a task that doesn't have a newton preset), not another infrastructure / shape bug. If the infrastructure-level bug persists, capture the dispatch.json failure kind/message and treat it as a new finding to follow up on.

- [ ] **Step 6: Update the architecture doc**

Edit `docs/odin/architecture.md`:
- Change the T3 row in §6's task-map table. It currently has two specs listed; **append** the new integration-fixes spec to the list. Replace:
  ```
  | T3 | Distributed dispatcher (Layer 3) + Asgard | T3.1 `docs/superpowers/specs/2026-04-22-odin-t3-1-dispatch-design.md`; bootstrap `docs/superpowers/specs/2026-04-23-odin-bootstrap-design.md` | 🟡 |
  ```
  with:
  ```
  | T3 | Distributed dispatcher (Layer 3) + Asgard | T3.1 `docs/superpowers/specs/2026-04-22-odin-t3-1-dispatch-design.md`; bootstrap `docs/superpowers/specs/2026-04-23-odin-bootstrap-design.md`; integration fixes `docs/superpowers/specs/2026-04-24-odin-t3-1-integration-fixes-design.md` | 🟡 |
  ```
- Bump the "Last updated" line to `2026-04-24 (T3.1 integration fixes landed)`.
- Add a new row to §9 change log:
  ```
  | 2026-04-24 | T3.1 integration fixes landed. Closed three gaps that only surfaced on the first real-fleet dispatch attempt (spec: `docs/superpowers/specs/2026-04-24-odin-t3-1-integration-fixes-design.md`): (1) Hugin/Munin wrappers now accept a `--run_id` CLI override that bypasses `compute_run_id(...)`, so the dispatcher-computed id is used end-to-end; (2) `docker/docker-compose.yaml` bind-mounts `~/IsaacLab/odin_runs` into the isaac-lab-base container at `/workspace/isaaclab/odin_runs`, so bundles written inside the container land on the host where T3.1's rsync-pull expects them; (3) `tools/odin/asgard/worker.py:_build_docker_exec_cmd` appends `--run_id <job.run_id>` to every Hugin/Munin invocation. Bootstrap also grew a `create_odin_runs` step to pre-create the bind-mount source (owned by the SSH user, not root-via-docker). Real-fleet smoke (4 jobs, 1 task × 2 seeds × 2 backends) now completes end-to-end. | Odin T3.1 integration fixes |
  ```

- [ ] **Step 7: Pre-commit + commit**

```bash
./isaaclab.sh -f
git add docs/odin/architecture.md
git commit -m "Mark T3.1 integration fixes in architecture doc"
```

Subject is 48 chars.

---

## Summary of verification criteria

After all 5 tasks the following must hold:

- `./isaaclab.sh -p -m pytest tools/odin/tests/ scripts/benchmarks/tests/` → all pass, including the 4 new tests (2 Hugin/Munin `--run_id` overrides, 2 bootstrap `create_odin_runs`) and the updated worker assertion.
- `./isaaclab.sh -f` → clean.
- `odin-bootstrap --fleet fleet.yaml` against both Valkyries → `2/2 hosts ok`, with `docker inspect` showing the `odin_runs` bind-mount.
- `odin-dispatch ... --include 'Isaac-Ant-Direct-v0' --seeds 42,43` → `4 completed, 0 failed`, every bundle populated, aggregate.json lands.
- Architecture doc reflects the new spec and the T3 row's updated link list.

## What comes next (out of scope for this plan)

- **Scale to the full T4.1 validation sweep**: 5 tasks × 3 seeds × 2 backends = 30 jobs. Takes ~1-2 hours. Run after this plan's smoke pass is green.
- **Any new gaps** that surface during the full sweep get their own small spec + plan cycle, per the pattern established in this session.
