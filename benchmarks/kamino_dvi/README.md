# Kamino DVI runtime benchmark

This package runs the approved RSL-RL Cartpole, Ant, and ANYmal-D comparison across current Kamino, Newton PR 3570 P-ADMM and
DVI, MJWarp, and PhysX. The matrix uses 300 training iterations and seeds 42–44. It starts at 4096 environments and
only moves down the declared capacity ladder after an explicit capacity failure.

## Locked environments

Create `.venv-current` from the project environment with Newton at the `newton_current` revision in `matrix.yaml`.
Create `.venv-pr3570` from the same environment, then install the exact PR revision without applying IsaacLab's
baseline package-source override:

```bash
uv pip install --python .venv-pr3570/bin/python --no-deps --reinstall --no-cache --no-config \
  "git+https://github.com/newton-physics/newton.git@7906676b2e5061273db96af179d7081fc6cbbba0"
```

For PhysX, `_isaac_sim` must point to a working Isaac Sim binary installation. The current/future runner validates exact Newton revisions, clean tracked IsaacLab files, and the configured IsaacLab/schema ancestry before executing any selected identity. New manifests record the exact clean IsaacLab HEAD and the matched TensorBoard event path and hash. The completed campaign predates those checks: its bundles record `versions.git_commit` and `versions.git_dirty`, but its manifests did not capture the runner-validated HEAD or retain and hash TensorBoard event files; the generated report discloses those limitations.

## Execute

Run five-iteration construction preflights first:

```bash
./isaaclab.sh -p -m benchmarks.kamino_dvi.run --preflight-only --resume
```

Then execute the full single-GPU matrix. Output is streamed to ignored per-run directories under
`benchmark_artifacts/kamino_dvi/runs`; atomic manifests make the command safe to resume.

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
  --decision-root benchmark_artifacts/kamino_dvi/decisions \
  --output benchmark_artifacts/kamino_dvi/decisions/stage2.json
./isaaclab.sh -p -m benchmarks.kamino_dvi.tune \
  --stage halve --artifact-root benchmark_artifacts/kamino_dvi/anymal_tuning \
  --decision-root benchmark_artifacts/kamino_dvi/decisions --resume
./isaaclab.sh -p -m benchmarks.kamino_dvi.analyze_tuning promote-finalists \
  --artifact-root benchmark_artifacts/kamino_dvi/anymal_tuning --logs-root logs \
  --decision-root benchmark_artifacts/kamino_dvi/decisions \
  --output benchmark_artifacts/kamino_dvi/decisions/finalists.json
./isaaclab.sh -p -m benchmarks.kamino_dvi.tune \
  --stage final --artifact-root benchmark_artifacts/kamino_dvi/anymal_tuning \
  --decision-root benchmark_artifacts/kamino_dvi/decisions --resume
./isaaclab.sh -p -m benchmarks.kamino_dvi.analyze_tuning select-winner \
  --artifact-root benchmark_artifacts/kamino_dvi/anymal_tuning --logs-root logs \
  --decision-root benchmark_artifacts/kamino_dvi/decisions \
  --output benchmark_artifacts/kamino_dvi/decisions/winner.json
./isaaclab.sh -p -m benchmarks.kamino_dvi.analyze_tuning report \
  --artifact-root benchmark_artifacts/kamino_dvi/anymal_tuning --logs-root logs \
  --decision-root benchmark_artifacts/kamino_dvi/decisions \
  --output-dir benchmarks/kamino_dvi/results/anymal_d_tuning
```

Stage 2 derives an immutable view of iterations 1--100 from each validated
300-iteration clean baseline, so its final-20 window is iterations 81--100.
After committing the selected configuration to the preset, validate that
committed preset without overrides:

```bash
./isaaclab.sh -p -m benchmarks.kamino_dvi.tune \
  --stage canonical --artifact-root benchmark_artifacts/kamino_dvi/anymal_tuning \
  --decision-root benchmark_artifacts/kamino_dvi/decisions --resume
```
