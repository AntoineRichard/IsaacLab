# Odin — Evaluation Harness (In-Tree)

Codename for the multi-backend IsaacLab evaluation harness. See the
[living architecture reference](../../docs/odin/architecture.md) for the
cross-task overview.

This directory currently lives inside IsaacLab for development convenience.
When Odin graduates, this whole directory moves to its own repo; the
IsaacLab-side benchmark scripts (`scripts/benchmarks/benchmark_*.py` and
`source/isaaclab/isaaclab/test/benchmark/standard_schema.py`) stay in place
and remain independently usable.

## Components

- `hugin/` — RSL-RL benchmark runner wrapper.
- `munin/` — SKRL benchmark runner wrapper.
- `common/` — shared helpers (run_id format, manifest writer, log tail).
- `tests/` — unit and integration tests (run without Kit runtime).

## Running a single bundle locally

```bash
# RSL-RL on PhysX
PYTHONPATH=. ./isaaclab.sh -p tools/odin/hugin/run.py \
    --task Isaac-Ant-Direct-v0 --backend physx --seed 42 \
    --num_envs 4096 --max_iterations 500

# SKRL on Newton
PYTHONPATH=. ./isaaclab.sh -p tools/odin/munin/run.py \
    --task Isaac-Ant-Direct-v0 --backend newton --seed 42 \
    --num_envs 4096 --max_iterations 500
```

Outputs land under `./odin_runs/<run_id>/` by default. See
[the spec](../../docs/superpowers/specs/2026-04-22-odin-t1-evaluation-runner-design.md)
for the bundle layout and schema.

## Running tests

```bash
./isaaclab.sh -p -m pytest tools/odin/tests/ -v --confcutdir=tools/odin
```

The `--confcutdir` flag bypasses the project-level `tools/conftest.py`,
which is written for IsaacLab's main test collection pipeline and is not
applicable here.

## Enumerating environments (T2.1)

T2.1 produces three committed artifacts that feed T3's dispatcher:

- `tools/odin/config/physx_envs.yaml` — curated PhysX run list.
- `tools/odin/config/newton_envs.yaml` — curated Newton run list (derived
  from the PhysX kept set that also has a `newton` preset).
- `docs/odin/newton_api_gaps.md` — narrative on what Newton is missing to
  unlock the remaining PhysX-kept tasks, plus a per-env appendix.

### Generate / refresh the PhysX list

Run from the repo root. `PYTHONPATH=.` makes `tools.odin.*` importable.

```bash
PYTHONPATH=. ./isaaclab.sh -p tools/odin/scripts/enumerate_physx_envs.py
# Options:
#   --output-path PATH   (default: tools/odin/config/physx_envs.yaml)
#   --dry-run            (print summary, write nothing)
#   --regenerate --force (discard existing YAML)
```

The script walks `gym.registry` for every `Isaac*` task, populates
`framework` / `num_envs` / `max_iterations` from the shipped framework
config, and writes the YAML grouped by directory-derived type. On re-run
it preserves your manual edits (`keep`, `framework`, `notes`, etc.) — rows
that vanish from the registry are kept with `status: stale`; new rows are
`status: new`.

### Curate the PhysX list

Edit `tools/odin/config/physx_envs.yaml` directly. Flip `keep: false` on
rows you don't want T3 to dispatch; adjust `framework` where the auto-pick
is wrong (e.g. force `skrl` on a vision task); tune `num_envs` /
`max_iterations` if the shipped defaults are wildly off for benchmarking.

### Generate the Newton list + gap candidates

```bash
PYTHONPATH=. ./isaaclab.sh -p tools/odin/scripts/enumerate_newton_envs.py
```

Reads your filtered `physx_envs.yaml`, writes:

- `tools/odin/config/newton_envs.yaml` for tasks whose env cfg declares a
  `newton` physics preset.
- `tools/odin/config/newton_gap_candidates.yaml` for the rest, each row
  carrying `suspected_gap: tbd`.

### Categorize the gap candidates and write the gap doc

1. Edit `tools/odin/config/newton_gap_candidates.yaml`: replace each
   `suspected_gap: tbd` with one of the controlled-vocabulary values
   (see `GAP_VOCABULARY` in `tools/odin/common/env_list.py`):

   - `preset_missing` — Newton would support this env, the task just
     needs its `newton` preset wired up. Not a physics gap.
   - `sdf_collision` — SDF colliders (rough terrain, nut-and-bolt, …).
   - `tendons` — tendon actuation.
   - `rough_terrain` — heightfield / procedural terrain (non-SDF).
   - `manipulation_coverage` — manipulation scene untested on Newton.
   - `deformable` — cloth / softbody / FEM assets.
   - `parallel_joints` — closed-loop or parallel kinematic constraints.
   - `controller_untested` — low-level controller stack untested on Newton.
   - `other` — doesn't fit the above; `notes:` required.

   `write_env_list` rejects any value outside the vocabulary on write.
   Use `notes:` to record secondary observations when one category
   isn't enough (e.g. a row that needs *both* `sdf_collision` and
   `rough_terrain` can be classified as the primary with the
   secondary noted).
2. Author `docs/odin/newton_api_gaps.md` with per-gap body sections
   (what's missing, count of affected envs, unlock value) followed by a
   per-env appendix table.

## Bootstrapping a fresh fleet

Fresh Valkyries — machines with only SSH, Docker, and a GPU — need to be
brought to T3.1-preflight-ready state before `odin-dispatch` will run
against them. `odin-bootstrap` handles this:

```bash
PYTHONPATH=. ./isaaclab.sh -p tools/odin/asgard/bootstrap_cli.py \
    --fleet fleet.yaml \
    [--build-timeout 1800] \
    [--sequential] \
    [--verbose]
```

For each host in `fleet.yaml`:

1. SSH reach + Docker daemon probes (fast).
2. `rm -rf {isaaclab_path}` (always — keeps re-runs idempotent).
3. `rsync` the controller's working tree to `{isaaclab_path}`.
4. `./docker/container.py start` with `--build-timeout` (default 1800 s
   = 30 min — covers a first-time docker image build).
5. `docker inspect` must report the container as `running`.

Hosts bootstrap in parallel by default (one thread per host); pass
`--sequential` if shared network bandwidth can't support simultaneous
rsyncs. Exit code is `0` iff every host succeeded.

After `odin-bootstrap` returns green, the T3.1 dispatch path works:

```bash
PYTHONPATH=. ./isaaclab.sh -p tools/odin/asgard/cli.py \
    --fleet fleet.yaml \
    --physx-yaml tools/odin/config/physx_envs.yaml \
    --seeds 42
```

Re-bootstrapping wipes and re-rsyncs — useful after long host idle (image
cache evicted), container drift, or as a belt-and-braces maintenance
pass. Don't re-bootstrap mid-dispatch: stop the dispatch first.

## Validating CUDA across the fleet

Newton (warp) workloads need at least CUDA 12.4 advertised by the host
NVIDIA driver. `odin-cuda` checks every Valkyrie and (optionally)
upgrades hosts that fall below the floor.

```bash
# Read-only: prints a per-host driver/cuda/status table, exits 1 if any
# host is below floor (default 12.4).
PYTHONPATH=. ./isaaclab.sh -p tools/odin/asgard/cuda_install_cli.py check \
    --fleet fleet.yaml

# Upgrade hosts below floor to cuda-12-9 (driver 575 + toolkit 12.9).
# Reboots each host. Prompts before doing anything; pass --yes to skip
# the prompt, --force to override the running-dispatch guard.
PYTHONPATH=. ./isaaclab.sh -p tools/odin/asgard/cuda_install_cli.py install \
    --fleet fleet.yaml --target 12.9
```

`install` runs the full pipeline per host: stop container → add NVIDIA
apt repo → `apt-get install cuda-12-9` → `systemctl reboot` → wait for
SSH → re-run `nvidia-smi` to verify driver family + CUDA floor → restart
container. Hosts already at-or-above floor are skipped without rebooting.

## Dispatching across a fleet (T3.1 — Asgard)

`tools/odin/asgard/cli.py` (the `odin-dispatch` entry point) ingests
the curated T2.1 YAMLs + a `fleet.yaml` listing SSH-accessible Valkyrie
machines and runs Hugin/Munin jobs across them in parallel.

### Fleet configuration

Create a `fleet.yaml` with per-host SSH / path config:

```yaml
fleet_name: h100-sweep-2026-04
default_ssh_user: odinrunner
default_ssh_key: ~/.ssh/odin_id_ed25519
hosts:
  - host: valkyrie-01.internal
  - host: valkyrie-02.internal
    ssh_user: svc-odin
    isaaclab_path: /mnt/scratch/IsaacLab
  - host: 10.0.0.42
    ssh_key: ~/.ssh/alt_key
```

Per-host fields override the fleet-level defaults. `container_name`
defaults to `isaac-lab-base` (matching `docker/docker-compose.yaml` for
profile `base`); override per-host if you're using a different profile.

### Running a dispatch

Run from the repo root. `PYTHONPATH=.` makes `tools.odin.*` importable.

```bash
PYTHONPATH=. ./isaaclab.sh -p tools/odin/asgard/cli.py \
    --fleet fleet.yaml \
    --physx-yaml tools/odin/config/physx_envs.yaml \
    --newton-yaml tools/odin/config/newton_envs.yaml \
    --seeds 42,43,44 \
    [--include 'Isaac-Ant-*'] \
    [--resume LATEST] \
    [--fresh] \
    [--skip-preflight] \
    [--per-job-timeout 14400] \
    [--verbose]
```

Each `keep: true` row in the YAML is expanded across the seed list into
one job per `(task, seed)`. Jobs dispatch concurrently — one per
Valkyrie. Bundles land in `odin_runs/<dispatch_id>/<run_id>/` on the
controller (rsync'd back on each job completion).

### What happens on first contact

For each Valkyrie in the fleet:

1. Preflight: SSH-reach + `docker ps` + `docker inspect <container_name>`
   + `test -d <isaaclab_path>`. A failure aborts the dispatch with a
   per-host report (`preflight.json` written either way). Use
   `--skip-preflight` to continue with the healthy hosts.
2. Provision: rsync the controller's working tree to the Valkyrie's
   `isaaclab_path`, then `./docker/container.py start` (or stop+start
   under `--fresh`). `--fresh` wipes the remote tree first.
3. Dispatch loop: pull a job from the shared queue, SSH in and
   `docker exec` Hugin/Munin with the job's CLI args, tee stdout to
   `<run_id>/logs/ssh-tail.log`, rsync the bundle back on exit,
   update `dispatch.json`.

### Failure classification

Jobs fail into one of four kinds (stored in `dispatch.json` under
`failure.kind`):

- `infrastructure` — SSH error / docker exec error before Hugin
  started. Retried up to `--max-infrastructure-retries` (default 2),
  preferring a different Valkyrie after the first failure.
- `hugin_crash` — remote process exited non-zero. **Not retried** —
  real bugs repeat; use `--retry-failed <run_id>` to explicitly
  re-attempt on a later invocation.
- `hugin_malformed_bundle` — Hugin exited 0 but `manifest.json` is
  missing / bad / wrong schema. Not retried.
- `timeout` — job ran past `--per-job-timeout`. Not retried; the
  remote process is terminated.

### Resume

If the controller crashes mid-dispatch, re-invoke with
`--resume <dispatch_id>` (or `--resume LATEST` to pick the most recent
directory). In-flight jobs flip back to `pending` and re-dispatch;
completed and failed jobs are preserved.

Starting a fresh dispatch means **not** passing `--resume` — a new
`<dispatch_id>` directory is created.

### State on disk

```
odin_runs/
└── 20260422-220000/                         # dispatch_id
    ├── dispatch.json                         # full state, atomically rewritten
    ├── fleet.yaml.snapshot                   # fleet.yaml at dispatch start
    ├── preflight.json                        # opening health check
    ├── aggregate.json                        # per-dispatch rollup (T4.1)
    ├── rsl-rl_physx_Isaac-Ant-Direct-v0_20260422-220000_seed42/
    │   ├── manifest.json
    │   ├── training.json
    │   ├── startup.json
    │   ├── training_data/              # TB events + params/ + checkpoints, written by the training framework
    │   └── logs/
    │       ├── ssh-tail.log                  # controller-side tee of remote stdout
    │       └── (Hugin's own log files)
    └── ...
```

`dispatch.json` schema v1.0 is defined in
`tools/odin/asgard/state.py`. See
`docs/superpowers/specs/2026-04-22-odin-t3-1-dispatch-design.md` for
field-by-field details.

## Aggregating a dispatch (T4.1 — Valhalla)

Every dispatch auto-produces `odin_runs/<dispatch_id>/aggregate.json` at
the tail of `run_dispatch` (opt out with `--skip-aggregate`). The
aggregate rolls every bundle into a nested `(task, framework, backend) →
seeds{} + aggregate{}` shape with cross-seed mean/std/min/max/cv_pct on
six headline metrics (`reward_final_ema`, `ep_length_final_ema`,
`iter_time_s_mean`, `env_steps_per_s_mean`, `ram_gb_peak`,
`gpu_mem_gb_peak`). Seeds whose reward deviates more than `2.0 * std`
from the cross-seed mean land in `divergent_seeds[]`. Failed bundles
(or ones that fail the strict-whitelist validation) go in a top-level
`failures[]` with `failure_kind` from T3.1's classification plus the
synthesised `missing_bundle` / `malformed_bundle`.

### Re-running the aggregator manually

```bash
PYTHONPATH=. ./isaaclab.sh -p tools/odin/valhalla/cli.py <dispatch_id|LATEST> \
    [--runs-root odin_runs/] \
    [--divergence-z 2.0] \
    [--no-overwrite] \
    [--quiet]
```

Useful after `--retry-failed` lands new bundles in a prior dispatch, or
to aggregate a partial (in-flight) dispatch.

`aggregate.json` schema v1.0 and the full field list live in
`docs/superpowers/specs/2026-04-23-odin-t4-1-valhalla-aggregator-design.md`.
