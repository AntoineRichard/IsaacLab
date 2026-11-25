# Track Step Touch with the G1 and two pompons

This example trains and plays a SuperMimic motion-tracking policy for the handless 29-DoF Unitree G1. Each pompon has 32 seven-link guides. The hub and guide roots attach to the wrist with fixed joints in the same Newton VBD model as the robot. Each guide carries 16 skinned ribbons for a total of 512 visible ribbons per pompon. OvRTX renders the ribbons in six shades of green with plastic and foil materials.

| Setting | Value |
| --- | --- |
| Newton | `d31603ff5e8389d1d6b2a5daef6ea2cb074c26ff`, local per-body rigid solve |
| Physics step | 2.5 ms (400 Hz) |
| Solver iterations | 12 per physics step |
| Joint-position target updates | Every 8 physics steps (50 Hz) |
| Floor shape stiffness | 100,000 N/m |
| Floor friction | 0.6 static and dynamic |

The G1 task comes from [Isaac Lab #9151](https://github.com/isaac-sim/IsaacLab/pull/9151) at `6579a7021b32f7603ba614977ab3904bc2b92cd2`. The example applies its solver and scene settings in the demo process. The [G1 USD](https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/Isaac/IsaacLab/Robots/Unitree/G1/g1_29dof.usd) is resolved by Isaac Lab's standard asset loader.

## Set up

Use a Linux CUDA system with the repository's [uv prerequisites](../../../docs/source/refs/contributing.rst). Run the commands from the repository root. Git LFS is needed to fetch the reference motion.

```bash
git lfs pull --include="scripts/demos/g1_pompons/step_touch_50hz.npz" --exclude=""
uv sync --frozen --extra rsl-rl

git clone https://github.com/leggedrobotics/rsl_rl.git /tmp/g1-pompons-rsl-rl
git -C /tmp/g1-pompons-rsl-rl checkout 1f368fd2817ac9352e4836447cf73588b782bbae
export PYTHONPATH="/tmp/g1-pompons-rsl-rl${PYTHONPATH:+:$PYTHONPATH}"
```

The source overlay supplies the SuperMimic implementation from [RSL-RL #255](https://github.com/leggedrobotics/rsl_rl/pull/255). The Isaac Lab lockfile supplies Newton and the other training dependencies.

## Train

This command starts a fresh run with the bundled 32.48-second reference. The demonstration run used 4,096 environments on one L40S for four hours. Use `--num_envs 8 --max_iterations 1` for a quick launch check. Change the reference, if needed, with the Hydra argument `env.commands.motion.motion_file=/path/to/motion.npz`.

```bash
uv run --no-sync python scripts/demos/g1_pompons/g1_shared_vbd.py train \
  --rl_library rsl_rl --task Isaac-Motion-Tracking-G1 \
  --agent rsl_rl_supermimic_cfg_entry_point --num_envs 4096 --max_iterations 100000 \
  --seed 42 --device cuda:0 --visualizer none presets=state_obs \
  agent.experiment_name=g1_step_touch_pompons agent.save_interval=600
```

The learner adds corrections to reference joint-position targets and collects experience in the same VBD scene used for playback. RSL-RL saves checkpoints under `logs/rsl_rl/g1_step_touch_pompons`.

## Play or render

Supply a checkpoint produced by the training command. Playback starts at frame zero, allows the full reference duration, and retains the tracking termination checks. The demonstration checkpoint `model_19200.pt` reached `motion_end` at step 1,624 with 4.91 cm mean aligned body-position error.

```bash
uv run --no-sync python scripts/demos/g1_pompons/g1_shared_vbd.py \
  --checkpoint /path/to/model_19200.pt
```

For OvRTX, install the versions used for the demo in a separate directory and add them to the Python path. Use an empty output directory for each render.

```bash
uv pip install --no-deps --target /tmp/g1-pompons-rtx \
  ovrtx==0.6.0.1791947691 ovstage==0.3.0.1791986420
export PYTHONPATH="/tmp/g1-pompons-rtx:$PYTHONPATH"

uv run --no-sync python scripts/demos/g1_pompons/g1_shared_vbd.py \
  --checkpoint /path/to/model_19200.pt --render-frames /tmp/g1-pompons-frames --follow-camera

ffmpeg -framerate 30 -i /tmp/g1-pompons-frames/frame_%04d.png \
  -c:v libx264 -pix_fmt yuv420p -movflags +faststart g1-pompons.mp4
```

## Motion credit

`step_touch_50hz.npz` adapts *Step Touch*, performed by Jasmine Coro, from the [G1 Moves dataset by Experiential Technologies](https://huggingface.co/datasets/exptech/g1-moves/tree/main/dance/J_Dance0_StepTouch) under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). We resampled the retargeted motion to 50 Hz and converted it to the Isaac Lab motion-tracking NPZ format.
