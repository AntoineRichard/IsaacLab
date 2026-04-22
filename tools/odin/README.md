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
   `suspected_gap: tbd` with one of
   `sdf_collision | tendons | rough_terrain | manipulation_coverage | deformable | other`.
   Use `notes:` to add context where it helps the gap narrative.
2. Author `docs/odin/newton_api_gaps.md` with per-gap body sections
   (what's missing, count of affected envs, unlock value) followed by a
   per-env appendix table.
