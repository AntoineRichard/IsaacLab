# Kamino DVI runtime benchmark

This package runs the approved RSL-RL Cartpole, Ant, and ANYmal-D comparison across current Kamino, Newton PR 3570
P-ADMM and DVI, MJWarp, and PhysX. Fourbar Pole and DR Legs add closed-loop coverage across the three Kamino variants
only. The 21-cell matrix uses 300 training iterations and seeds 42–44. It starts at 4096 environments and only moves
down the declared capacity ladder after an explicit capacity failure.

## Locked environments

Create `.venv-current` from the project environment with Newton at the `newton_current` revision in `matrix.yaml`.
Create `.venv-pr3570` from the same environment, then install the exact PR revision without applying IsaacLab's
baseline package-source override:

```bash
uv pip install --python .venv-pr3570/bin/python --no-deps --reinstall --no-cache --no-config \
  "git+https://github.com/newton-physics/newton.git@7906676b2e5061273db96af179d7081fc6cbbba0"
```

For PhysX, `_isaac_sim` must point to a working Isaac Sim binary installation. The current/future runner validates exact Newton revisions, clean tracked IsaacLab files, and the configured IsaacLab/schema ancestry before executing any selected identity. New manifests record the exact clean IsaacLab HEAD and the matched TensorBoard event path and hash. The completed campaign predates those checks: its bundles record `versions.git_commit` and `versions.git_dirty`, but its manifests did not capture the runner-validated HEAD or retain and hash TensorBoard event files; the generated report discloses those limitations.

Training commands prepend the launched checkout's `source/isaaclab_newton` and
`source/isaaclab_tasks` directories to the child `PYTHONPATH`. Before GPU work,
the runner probes that exact environment and rejects an `isaaclab_newton`
import outside the launched checkout. Tuning manifest schema 1.2 persists the
resolved module path, distribution location, and `direct_url.json` metadata.
Schema 1.1 tuning manifests are intentionally not resumable as schema 1.2
evidence; start a new artifact root when migrating a campaign.

## Execute

Run five-iteration construction preflights first:

```bash
./isaaclab.sh -p -m benchmarks.kamino_dvi.run --preflight-only --resume
```

Then execute the full single-GPU matrix. Output is streamed to ignored per-run directories under
`benchmark_artifacts/kamino_dvi/runs`; atomic manifests make the command safe to resume.

When `--resume` finds an existing generic-run manifest, the runner preserves it
and its logs unless the identity, command, locked revisions, schema, artifact
location, and IsaacLab HEAD are compatible. Unreadable or incompatible
manifests—including manifests from the earlier child-`PYTHONPATH` command—raise
an error before execution and require a new artifact root.

```bash
./isaaclab.sh -p -m benchmarks.kamino_dvi.run --full-only --resume
```

Use `--task`, `--variant`, and `--seed` to select a subset. `--dry-run` prints the exact commands without probing or
launching an environment.

## Analyze

Generate compact JSON, 95% CI runtime and learning figures, and Markdown/PDF reports:

```bash
./isaaclab.sh -p -m benchmarks.kamino_dvi.analyze \
  --artifact-root benchmark_artifacts/kamino_dvi/runs \
  --logs-root logs \
  --output-dir benchmarks/kamino_dvi/results
```

Runtime summaries exclude iterations 1–10. Reward, episode length, and success summaries average the final 20
iterations. Confidence intervals use the two-sided three-seed Student-t critical value. Reward and episode length are
read from schema v1.1; success is read from the matching TensorBoard trace because the original v1.1 generator stored
live step averages instead of the logged per-iteration values. That generator bug is fixed separately in PR 6624.

## ANYmal-D task-specific tuning

The tuning campaign always uses 4096 environments. Run each measured stage only
after its candidate preflights, and create the named decision before starting
the next adaptive stage. The analyzer rejects incomplete identity coverage,
reduced counts, nonfinite data, and mismatched evidence provenance.

Adaptive decision schema 1.1 includes package-location provenance. Before a
`wave2`, `halve`, `final`, or `canonical` production launch, the runner uses the
strict analyzer to recompute the complete persisted decision chain from raw
evidence. Any changed selection, configuration, status, revision, source hash,
rejection provenance, or upstream decision stops before source probing or GPU
execution. Report schema 1.2 labels run counts separately from candidate
learning rejections and promotions.

For Wave 1 and Wave 2 seed 42, the analyzer also validates the exact
five-iteration screening preflight for each expected candidate. When the
latest contiguous preflight attempt failed and no measured record exists, the
analyzer projects that immutable failure in memory as a rejected tuning record
using the actual preflight run ID and a `preflight:<category>` reason. It does
not write or backfill artifacts. A completed preflight does not substitute for
a missing measurement, measured evidence always wins, and preflights never
synthesize seed 43/44 or later-stage evidence. Decisions, validation output,
summary JSON, Markdown, and PDF reports disclose these derived rejection sources.

Validate any completed stages without mutating evidence before advancing the
campaign. Baseline and Wave 1 validation require no decision files:

```bash
./isaaclab.sh -p -m benchmarks.kamino_dvi.analyze_tuning validate \
  --stages baseline wave1 \
  --artifact-root benchmark_artifacts/kamino_dvi/anymal_tuning \
  --logs-root logs
```

The command writes deterministic standard JSON to stdout with expected,
terminal, valid, and rejected counts plus sorted run IDs, rejection reasons,
and the broad bundle-dirty count and run IDs. It exits nonzero without success
JSON when requested coverage or provenance is invalid. Adaptive stages read
their strict upstream decisions from `--decision-root` when supplied; otherwise
all staged analysis actions derive the shared directory from `--output`.

The runner enforces launch cleanliness with tracked files only (`git status
--porcelain --untracked-files=no`). The training bundle's
`versions.git_dirty` uses plain `git status --porcelain`, so it is a broader,
advisory flag that also includes untracked paths. Decisions, validation JSON,
and reports disclose every `true` bundle flag and its run ID. A `true` value
does not prove that only untracked paths differed; interpret it alongside the
runner's separate tracked-only check.

```bash
./isaaclab.sh -p -m benchmarks.kamino_dvi.tune \
  --stage baseline --artifact-root benchmark_artifacts/kamino_dvi/anymal_tuning --resume
./isaaclab.sh -p -m benchmarks.kamino_dvi.tune \
  --stage wave1 --artifact-root benchmark_artifacts/kamino_dvi/anymal_tuning --resume
./isaaclab.sh -p -m benchmarks.kamino_dvi.analyze_tuning resolve-wave2 \
  --artifact-root benchmark_artifacts/kamino_dvi/anymal_tuning --logs-root logs \
  --output benchmark_artifacts/kamino_dvi/decisions/wave2.json
./isaaclab.sh -p -m benchmarks.kamino_dvi.tune \
  --stage wave2 --artifact-root benchmark_artifacts/kamino_dvi/anymal_tuning \
  --decision-root benchmark_artifacts/kamino_dvi/decisions --resume
./isaaclab.sh -p -m benchmarks.kamino_dvi.analyze_tuning promote-stage2 \
  --artifact-root benchmark_artifacts/kamino_dvi/anymal_tuning --logs-root logs \
  --output benchmark_artifacts/kamino_dvi/decisions/stage2.json
./isaaclab.sh -p -m benchmarks.kamino_dvi.tune \
  --stage halve --artifact-root benchmark_artifacts/kamino_dvi/anymal_tuning \
  --decision-root benchmark_artifacts/kamino_dvi/decisions --resume
./isaaclab.sh -p -m benchmarks.kamino_dvi.analyze_tuning promote-finalists \
  --artifact-root benchmark_artifacts/kamino_dvi/anymal_tuning --logs-root logs \
  --output benchmark_artifacts/kamino_dvi/decisions/finalists.json
./isaaclab.sh -p -m benchmarks.kamino_dvi.tune \
  --stage final --artifact-root benchmark_artifacts/kamino_dvi/anymal_tuning \
  --decision-root benchmark_artifacts/kamino_dvi/decisions --resume
./isaaclab.sh -p -m benchmarks.kamino_dvi.analyze_tuning select-winner \
  --artifact-root benchmark_artifacts/kamino_dvi/anymal_tuning --logs-root logs \
  --output benchmark_artifacts/kamino_dvi/decisions/winner.json
```

If every Stage-2 candidate violates a per-seed learning guardrail,
`promote-finalists` writes a terminal `finalists.json` with no selected
candidates, every rejection reason, and status `stopped_no_safe_finalist`.
This is a valid scientific early stop: do not run `final`, select a winner,
modify the preset, or run `canonical`. The `final` runner and `select-winner`
both refuse that empty decision.

Run the `report` command directly after the early stop. It validates all
available evidence and emits the same exact five files. The report labels
Wave 1/2 as single-seed observations, gives two-sided 95% Student-t intervals
for the seed-42/43 Stage-2 metrics, and treats legacy MJWarp/PhysX runtimes as
context only; it does not attribute a speedup to a winner.

Stage 2 derives an immutable view of iterations 1--100 from each validated
300-iteration clean baseline, so its final-20 window is iterations 81--100.
After winner selection, update the literal solver preset with the winner and
commit that preset. Then validate the committed preset without overrides:

```bash
./isaaclab.sh -p -m benchmarks.kamino_dvi.tune \
  --stage canonical --artifact-root benchmark_artifacts/kamino_dvi/anymal_tuning \
  --decision-root benchmark_artifacts/kamino_dvi/decisions --resume
```

Only after the three-seed canonical validation passes, generate the report:

```bash
./isaaclab.sh -p -m benchmarks.kamino_dvi.analyze_tuning report \
  --artifact-root benchmark_artifacts/kamino_dvi/anymal_tuning --logs-root logs \
  --decision-root benchmark_artifacts/kamino_dvi/decisions \
  --output-dir benchmarks/kamino_dvi/results/anymal_d_tuning
```

The report directory contains exactly `summary.json`, `runtime.png`,
`learning.png`, `anymal_d_dvi_tuning.md`, and `anymal_d_dvi_tuning.pdf`.
