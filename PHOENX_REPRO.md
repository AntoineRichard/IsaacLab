# PhoenX in IsaacLab — Reproduction Guide (kit-less)

**Branch:** `antoiner/feature/phoenx-solver` on `https://github.com/AntoineRichard/IsaacLab.git`

**Clone:**
```bash
git clone -b antoiner/feature/phoenx-solver https://github.com/AntoineRichard/IsaacLab.git
```

This branch depends on a **local Newton checkout that contains `SolverPhoenX`** (the upstream pinned SHA in `isaaclab_newton/setup.py` does not have it yet).

---

## 1. Clone IsaacLab and Newton

```bash
mkdir -p ~/work && cd ~/work
git clone -b antoiner/feature/phoenx-solver https://github.com/AntoineRichard/IsaacLab.git IsaacLab2
git clone https://github.com/newton-physics/newton.git newton_2
# newton_2 must contain newton._src.solvers.phoenx — main does as of 2026-05-06.
```

## 2. Create a kit-less uv venv (Python 3.12)

```bash
cd ~/work/IsaacLab2
uv venv --python 3.12 .venv
source .venv/bin/activate
```

If step 3 fails on `setuptools<82.0.0` with a 404 from `pypi.nvidia.com`, run this once and retry:

```bash
UV_EXTRA_INDEX_URL=https://pypi.nvidia.com uv pip install 'setuptools<82.0.0'
```

## 3. Install IsaacLab kit-less (newton + tasks + assets + RL + visualizers)

```bash
./isaaclab.sh -i 'newton,tasks,assets,rl[rsl_rl],visualizers'
```

Pulls torch+cu128, rsl-rl, the in-tree extensions, and a GitHub-pinned Newton.

## 4. Swap the GitHub-pinned Newton for the local checkout

```bash
uv pip install -e ../newton_2[sim]
```

Bumps `mujoco-warp` 3.6 → 3.7, pulls in `newton-actuators`, pins `warp-lang` to ≥1.13rc1.

## 5. Upgrade `warp-lang` to 1.14 dev

The committed `wp.math.transform_to_matrix` fix on this branch is one site; other kernels still want the 1.14-era APIs.

```bash
uv pip install --extra-index-url https://pypi.nvidia.com --prerelease=allow \
  --index-strategy unsafe-best-match 'warp-lang==1.14.0.dev20260506'
```

## 6. Side-install `isaaclab_physx`, `isaaclab_ovphysx`, `isaaclab_contrib` (no deps)

Some task configs (cartpole, G1) eagerly import these extensions even when their physics manager is never used. Install editably without dependencies — those would pull in Isaac Sim:

```bash
uv pip install --no-deps -e source/isaaclab_physx \
                          -e source/isaaclab_ovphysx \
                          -e source/isaaclab_contrib
```

If the install fails with `Cannot update time stamp of directory 'isaaclab_*.egg-info'` (root-owned stale egg-info from a prior root-mode install), `sudo rm -rf` those `.egg-info` directories under `source/isaaclab_*/` first.

## 7. Verify

```bash
./isaaclab.sh -p -m pytest source/isaaclab_newton/test/physics/test_newton_manager_abstraction.py -v
```

All 30+ tests should pass, including the new `phoenx`, `test_phoenx_with_collision_cfg_raises`, and `test_phoenx_velocity_readout_modes` cases.

## 8. Run training

```bash
# Cartpole — best PhoenX result, fastest
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Isaac-Cartpole-Direct-v0 --num_envs=4096 --max_iterations=150 \
  presets=newton_phoenx --headless

# Ant — tuned config (substeps=8 / sol_iter=16 / vel_iter=2 / substep_average)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Isaac-Ant-Direct-v0 --num_envs=4096 --max_iterations=300 \
  presets=newton_phoenx --headless

# G1 flat — contact sensor and dependent reward / termination terms auto-disabled
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Isaac-Velocity-Flat-G1-v0 --num_envs=4096 --max_iterations=200 \
  presets=newton_phoenx --headless
```

Use `presets=newton_mjwarp` for the MJWarp baseline.

---

## Known limitations

- **Humanoid** (`Isaac-Humanoid-Direct-v0`): not supported. The asset uses a `JointType.D6` joint that `SolverPhoenX` raises on. No preset shipped.
- **Contact sensors**: PhoenX has no contact-sensor backend on the IsaacLab Newton manager today. The G1 preset opts out by sniffing `presets=newton_phoenx` from `sys.argv` in `G1RoughEnvCfg.__post_init__` (preset resolution runs after `__post_init__`).
- **Setup time on G1**: PhoenX `Init solver` is ~10 s versus MJWarp ~2 s for the humanoid kinematic chain (ADBS column setup); PhoenX is faster than MJWarp on cartpole/ant.
- **G1 PhoenX value loss**: explodes to ~1e15 with `velocity_readout="substep_end"` on training-from-scratch runs (the Newton example uses that mode for *inference*, not training). Try `"substep_average"` if training from scratch on G1.

---

## Performance summary (4096 envs, headless)

| Task | Iters | PhoenX steady-state | MJWarp steady-state | Ratio | PhoenX setup | MJWarp setup |
|---|---:|---:|---:|---:|---:|---:|
| Cartpole | 150 | 593 k st/s | 729 k st/s | 0.81× | 1.2 s | 36.1 s |
| Ant | 300 | 426 k st/s | 449 k st/s | 0.95× | 9.5 s | 22.0 s |
| G1 (contact-disabled) | 200 | 82 k st/s | 133 k st/s | 0.62× | 16.9 s | 8.6 s |

PhoenX wins setup on small-to-medium robots, loses setup on G1. Steady-state throughput gap widens with joint count + contact density.

## Branch commits

```
dc44faee309 Add newton_phoenx preset to G1 flat locomotion
4105d6487dc Fix wp.math.transform_to_matrix removed in warp 1.14
4ee38e067a8 Tune Ant newton_phoenx preset for stability
7055da2b47d Add newton_phoenx preset to Ant
86df95ec2b8 Add newton_phoenx preset to Cartpole
4fedf5c6a4e Cover all three PhoenX velocity_readout modes with a smoke test
9dbde1cc2f5 Add regression test for PhoenX collision_cfg rejection
0ce0a7ece80 Implement NewtonPhoenXManager build_solver and contact adoption
0ae13e099e4 Add PhoenX solver config and manager skeleton
```
