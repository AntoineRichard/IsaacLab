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
