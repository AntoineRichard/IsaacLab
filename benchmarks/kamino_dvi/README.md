# Kamino DVI runtime benchmark

This package runs the approved RSL-RL comparison across current Kamino, Newton PR 3570 P-ADMM and DVI,
MJWarp, and PhysX. The matrix uses 300 training iterations and seeds 42–46. It starts at 4096 environments and
only moves down the declared capacity ladder after an explicit capacity failure.

## Locked environments

Create `.venv-current` from the project environment with Newton at the `newton_current` revision in `matrix.yaml`.
Create `.venv-pr3570` from the same environment, then install the exact PR revision without applying IsaacLab's
baseline package-source override:

```bash
uv pip install --python .venv-pr3570/bin/python --no-deps --reinstall --no-cache --no-config \
  "git+https://github.com/newton-physics/newton.git@7906676b2e5061273db96af179d7081fc6cbbba0"
```

For PhysX, `_isaac_sim` must point to a working Isaac Sim binary installation. The runner probes and validates both
Newton revisions before executing any selected identity.

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

Generate compact JSON, a 95% CI runtime figure, and Markdown/PDF reports:

```bash
./isaaclab.sh -p -m benchmarks.kamino_dvi.analyze \
  --artifact-root benchmark_artifacts/kamino_dvi/runs \
  --logs-root logs \
  --output-dir benchmarks/kamino_dvi/results
```

Runtime summaries exclude iterations 1–10. Reward, episode length, and success summaries average the final 20
iterations. Confidence intervals use the two-sided five-seed Student-t critical value. Reward and episode length are
read from schema v1.1; success is read from the matching TensorBoard trace because the original v1.1 generator stored
live step averages instead of the logged per-iteration values. That generator bug is fixed separately in PR 6624.
