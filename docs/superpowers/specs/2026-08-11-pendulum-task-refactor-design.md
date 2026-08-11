# Pendulum Task Refactor Design

**Date:** 2026-08-11

## Summary

Refactor the cart-double-pendulum task family into three explicit environments:

- `Isaac-Pendulum-Direct`: a new canonical single-agent `DirectRLEnv`.
- `Isaac-Pendulum`: a new canonical single-agent `ManagerBasedRLEnv`.
- `Isaac-Pendulum-MARL-Direct`: the existing cooperative `DirectMARLEnv`.

The new single-agent direct and manager-based environments must implement the
same Markov decision process: equivalent observations, actions, rewards,
resets, terminations, time-step scaling, and success metrics. The MARL task is
kept as a separate example and retains its existing agent-specific behavior.

This is an intentional breaking change. The current `Isaac-Pendulum-Direct`
MARL task moves immediately to `Isaac-Pendulum-MARL-Direct`; no deprecated task
alias or compatibility forwarding module will be provided. The changelog must
state that existing MARL users need to select the new task ID.

## Goals

1. Provide matching single-agent direct and manager-based double-pendulum
   tasks, following the cleaned cartpole task family.
2. Preserve the existing task as an explicit MARL example under a clear name.
3. Put reusable task mathematics behind small tensor-level functions so the
   single-agent workflows cannot drift semantically.
4. Improve the single-agent task definition where the existing MARL behavior
   is inconsistent, then validate the changes through training.
5. Keep workflow-specific environment code and configuration readable rather
   than hiding it behind a large shared abstraction.

## Non-goals

- Making the MARL task numerically equivalent to the single-agent tasks.
- Preserving old task IDs, module paths, class names, or checkpoints.
- Adding RSL-RL or Stable-Baselines3 configurations in this refactor.
- Performing speculative performance optimization before semantic correctness
  and trainability are established.
- Adding camera observations or visual task variants.

## Package structure

The package will follow the flat workflow-specific layout used by cartpole:

```text
source/isaaclab_tasks/isaaclab_tasks/core/pendulum/
├── __init__.py
├── pendulum_marl_env.py
├── pendulum_marl_env_cfg.py
├── pendulum_direct_env.py
├── pendulum_direct_env_cfg.py
├── pendulum_manager_env_cfg.py
├── mdp/
│   ├── __init__.py
│   ├── observations.py
│   ├── rewards.py
│   └── terminations.py
└── agents/
```

The current `pendulum_env.py` and `pendulum_env_cfg.py` modules will be renamed
rather than retained as compatibility wrappers. Public documentation and all
in-repository references must be updated to the new paths.

## Environment registrations

The package registers exactly these task IDs:

| Task ID | Environment type | Purpose |
|---|---|---|
| `Isaac-Pendulum-Direct` | `DirectRLEnv` | Canonical single-agent direct task |
| `Isaac-Pendulum` | `ManagerBasedRLEnv` | Canonical single-agent manager task |
| `Isaac-Pendulum-MARL-Direct` | `DirectMARLEnv` | Cooperative multi-agent example |

The old meaning of `Isaac-Pendulum-Direct` changes immediately from MARL to
single-agent. The major changelog fragment must provide this migration:

```text
Use Isaac-Pendulum-MARL-Direct to continue running the multi-agent task.
Use Isaac-Pendulum-Direct for the new single-agent direct task.
```

## Canonical single-agent MDP

Both workflows use the same `CART_DOUBLE_PENDULUM_CFG` asset, 4,096
environments with 4 m spacing, a simulation time step of 1/120 s,
decimation 2, and a five-second episode. The resulting policy time step is
1/60 s and a complete episode contains 300 control steps. Workflow-specific
scene construction must not change these values.

### State and observation

Let:

- `x` be cart displacement from its default position [m].
- `theta_1` be the upper-pole joint position relative to its default [rad].
- `theta_2` be the lower joint position relative to its default [rad].
- Dotted quantities be their corresponding joint velocities.

Both single-agent environments expose the same six-dimensional policy
observation, in positions-then-velocities order:

```text
[x, theta_1, theta_2, x_dot, theta_1_dot, theta_2_dot]
```

The direct environment constructs this tensor explicitly. The manager-based
environment concatenates equivalent position-relative and velocity-relative
observation terms in the same order. Observation corruption is disabled.

### Action

Both environments expose a two-dimensional continuous action:

```text
[cart_force, lower_joint_torque]
```

The first component drives `slider_to_cart` with a scale of 100 N. The second
drives `pole_to_pendulum` with a scale of 50 N·m. The upper
`cart_to_pole` joint remains passive. Manager action-term declaration order
must preserve this vector ordering.

### Kinematic task quantities

The upper-link angle and angular velocity are:

```text
upper_angle = wrap_to_pi(theta_1)
upper_velocity = theta_1_dot
```

The lower link is evaluated in the world-relative planar orientation implied
by the serial chain:

```text
lower_angle = wrap_to_pi(theta_1 + theta_2)
lower_velocity = theta_1_dot + theta_2_dot
```

The summed lower angle is wrapped after addition. This avoids the current MARL
implementation's discontinuity from wrapping each joint separately and then
adding the results.

### Reward

The single-agent reward rate is composed from one shared balancing objective
and the two actuator-local shaping groups discussed during design:

```text
shared = alive + terminating + upper_angle_position + lower_angle_position
cart_local = cart_velocity + upper_angular_velocity
pendulum_local = lower_absolute_angular_velocity
reward_rate = shared + cart_local + pendulum_local
reward = reward_rate * step_dt
```

Initial weights are:

| Term | Weight |
|---|---:|
| Alive while not terminated | `1.0` |
| Early termination | `-2.0` |
| Squared upper-link angle | `-1.0` |
| Squared lower-link absolute angle | `-1.0` |
| Absolute cart velocity | `-0.01` |
| Absolute upper-link angular velocity | `-0.01` |
| Absolute lower-link angular velocity | `-0.01` |

The direct implementation multiplies the summed reward rate by `step_dt`
explicitly. The manager implementation supplies the same weights to the
reward manager, which performs the `step_dt` multiplication. Neither workflow
may apply the factor twice.

Tensor-level helpers define the angle errors and unweighted reward components.
Manager reward terms extract the configured articulation data and delegate to
those helpers. The direct environment delegates to the same helpers.

### Reset distribution

Both single-agent environments reset from articulation defaults and sample:

| Quantity | Uniform range |
|---|---|
| Cart position | `[-1.0, 1.0]` m |
| Cart velocity | `[-0.5, 0.5]` m/s |
| Upper-pole angle | `[-0.25 pi, 0.25 pi]` rad |
| Upper-pole velocity | `[-0.25 pi, 0.25 pi]` rad/s |
| Lower-joint angle | `[-0.25 pi, 0.25 pi]` rad |
| Lower-joint velocity | `[-0.25 pi, 0.25 pi]` rad/s |

Sampled joint state is clamped to the articulation's soft position and
velocity limits. The direct implementation mirrors
`reset_joints_by_offset`; the manager implementation uses that event term.

### Termination and success

An episode terminates early when any of these conditions holds:

- `abs(x) > 3.0` m.
- `abs(wrap_to_pi(theta_1)) > pi / 2` rad.
- `abs(wrap_to_pi(theta_1 + theta_2)) > pi / 2` rad.

An episode times out after five seconds. A successful episode is one that
times out without early termination. Both workflows log
`Metrics/success_rate` with this definition.

The expanded reset distribution and lower-link termination are intentional
changes from the MARL task and are subject to the training fallback described
below.

## MARL environment

The existing environment moves to `pendulum_marl_env.py` and uses explicit
class names `PendulumMARLEnv` and `PendulumMARLEnvCfg`.

It retains:

- The `cart` and `pendulum` agents.
- One action per agent with the existing 100 N and 50 N·m scales.
- The existing four-dimensional cart observation.
- The existing three-dimensional pendulum observation.
- The existing separate per-agent rewards and weights.
- The existing reset distribution and termination behavior.
- The automatically concatenated MARL state used by MAPPO.

The refactor may reuse side-effect-free helpers only where doing so is exactly
behavior-preserving. In particular, it must not silently change the MARL angle
wrapping, reward scaling, observation ordering, reset distribution, or done
conditions merely to share more code.

Existing RL-Games PPO and skrl PPO/IPPO/MAPPO configurations remain attached to
`Isaac-Pendulum-MARL-Direct`. Their filenames gain a `marl` or `direct_marl`
qualifier where needed to distinguish them from single-agent configurations.

## Single-agent training configurations

The new direct and manager tasks receive distinct RL-Games PPO and skrl PPO
configuration files. Both workflows initially use the existing pendulum PPO
network and optimizer settings, with only environment-interface fields and
experiment names changed. Hyperparameters may diverge later only when
training evidence shows that a workflow requires it. Task registrations must
not point both workflows at one configuration file.

RSL-RL and Stable-Baselines3 support are deferred until the task semantics and
training behavior are stable.

## Reuse boundaries

Reuse is intentionally limited to stable task concepts:

- Tensor helpers for canonical observation assembly.
- Tensor helpers for upper- and lower-link absolute kinematics.
- Tensor helpers for reward components.
- Tensor helpers for single-agent termination predicates.

Environment lifecycle, scene setup, manager configuration declarations, and
MARL dictionary assembly remain workflow-specific. Avoid introducing a common
environment superclass or a broad mixin: the three IsaacLab base classes have
different contracts, and such inheritance would make the task harder to read.

## Testing

### Tensor-level tests

Test all shared task helpers without launching simulation:

- Canonical observation ordering.
- Lower absolute angle wrapping after joint-angle addition.
- Lower absolute angular velocity computation.
- Every reward component independently.
- Cart, upper-link, and lower-link termination predicates.
- Boundary behavior at the configured limits.

### Direct/manager parity tests

Inject or construct equivalent articulation states and verify:

- Identical six-dimensional policy observations.
- Identical unweighted reward components.
- Identical total rewards after accounting for the manager's automatic
  `step_dt` multiplication.
- Identical early-termination and timeout signals.
- Equivalent reset ranges and limit clamping.
- Identical success-rate meaning.

Tests should compare named reward components as well as totals so compensating
errors cannot hide a mismatch.

### MARL regression tests

Protect the retained MARL behavior during its file/class rename:

- Per-agent observation shapes and ordering.
- Per-agent action ownership and scaling.
- Per-agent reward formulas.
- Shared termination dictionaries.
- Existing reset ranges.
- MAPPO state construction.

### Repository verification

Run focused pendulum tests, task-registration tests, determinism/smoke tests,
and `./isaaclab.sh -f`. Because public files and symbols are added, regenerate
the documentation with `./isaaclab.sh -d` and review the generated changes.

## Training acceptance and fallback

Train both new single-agent variants with skrl PPO, the same initial
hyperparameters, and seed 42 for 300 training iterations. Each run must remain
numerically stable, show no simulator errors, and reach a reported mean
episode length of at least 150 control steps. Inspect success rate and the
reward curve as secondary evidence that improvement is sustained rather than
an isolated evaluation spike. Reward thresholds must be established from the
new time-scaled runs instead of carrying forward the MARL reward threshold.

Keep the existing MARL benchmark under `Isaac-Pendulum-MARL-Direct`. Add
benchmark entries for the new direct and manager tasks using thresholds
supported by the observed training results.

If the candidate task does not train reliably:

1. Remove only the new lower-link early-termination condition and retrain.
2. If needed, restore the narrower current reset distribution and retrain.
3. Preserve the package structure, direct/manager parity, canonical
   observation, reward decomposition, and correct time-step scaling.

Each fallback must be evaluated as a separate change so the cause is visible.

Performance work follows correctness and trainability. Profile the resulting
environments before considering caching or fusion of manager terms; do not add
complexity based solely on assumed overhead.

## Documentation and changelog

Update the environment table and any task links to show all three tasks and
their workflow types. Document that the single-agent pair is equivalent while
the MARL task intentionally has a different interface and reward structure.

Add one `isaaclab_tasks` major changelog fragment with migration guidance:

- `Isaac-Pendulum-Direct` now identifies the single-agent direct task.
- Existing MARL users must migrate to `Isaac-Pendulum-MARL-Direct`.
- Python imports must migrate to the renamed MARL modules and classes.
- `Isaac-Pendulum` is the new manager-based equivalent of the canonical direct
  task.

Do not edit `CHANGELOG.rst` or `config/extension.toml` directly.
