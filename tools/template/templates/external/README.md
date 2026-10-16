# {{ name }}

An installable downstream Isaac Lab task package with a standard uv `src` layout.

{% if specifications %}The registered tasks are:
{% for specification in specifications %}
- `{{ specification.task.id }}`
{% endfor %}
{% else %}No tasks are registered yet. Add task packages under `src/{{ name }}/tasks` and register them with Gymnasium.
{% endif %}

{% if initial_content == "stubbed" %}
## Implement the Task

This project contains task scaffolding. Before running or training, replace `MISSING` configuration values and
implement functions that raise `NotImplementedError`. Define the scene and robot assets, physics and timing,
action and observation spaces, rewards, termination conditions, and reset behavior. Manager-based task terms live
in the family's `mdp` package. Define assets in the scene configuration; direct environments inherit scene creation
from the framework. If AMP is selected, implement reference-motion collection as well. Direct AMP tasks must also
update observation history and supply `self.extras["amp_obs"]` on every step.

Registrations and agent configurations are provided so you can build on the same structure as the runnable example.
{% endif %}

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then create the project environment. The default
environment uses the Newton backend and does not install Isaac Sim:

```bash
uv sync
```
{% if isaaclab_extra_groups %}
{% set extra_kind = "group" %}
{% set extra_option = "--group" %}

The project mirrors every optional extra declared by the Isaac Lab checkout used to create it, including physics,
rendering, visualizer, RL, teleoperation, and test features, as a dependency group of the same name that installs the
extra from the checkout. Inspect `pyproject.toml` for the complete list and pass each required group to `uv run`
(the `--extra` options select the published extras, which do not install anything from the checkout):
{% else %}
{% set extra_kind = "extra" %}
{% set extra_option = "--extra" %}

The project forwards every optional extra declared by the Isaac Lab package used to create it, including physics,
rendering, visualizer, RL, teleoperation, and development features. Inspect `pyproject.toml` for the complete list and
pass each required extra to `uv run`:
{% endif %}

```bash
# Standalone OV PhysX
uv run {{ extra_option }} ovphysx isaaclab random_agent --task <TASK_NAME> physics=ovphysx

# Isaac Sim with PhysX and RTX rendering
uv run {{ extra_option }} isaacsim isaaclab random_agent --task <TASK_NAME> physics=isaacsim_physx

# Isaac Sim with the Rerun visualizer
uv run {{ extra_option }} isaacsim {{ extra_option }} rerun isaaclab random_agent --task <TASK_NAME> --viz rerun
```

The `all` {{ extra_kind }} forwards Isaac Lab's `all` selection. The `ov` {{ extra_kind }} installs both the `ovphysx`
and `ovrtx` runtimes; you can also select either independently. Commit `pyproject.toml` and `uv.lock` so collaborators
use the same environment.
{% if isaaclab_sources %}

### Develop against the Isaac Lab checkout

This project was generated from an Isaac Lab source checkout. Its published dependency is `{{ isaaclab_requirement }}`,
so built wheels work with released Isaac Lab packages, while the local environment uses the checkout:

- The `isaaclab-dev` dependency group installs the checkout through editable relative paths in `[tool.uv.sources]`.
  Update those paths if either directory moves.
- The peer overrides at the end of `[tool.uv] override-dependencies` resolve every `isaaclab*` requirement to the
  checkout, including the requirements of registry packages. uv therefore no longer checks their Isaac Lab version
  ranges; `isaaclab registry install` checks compatibility with Isaac Lab {{ isaaclab_release }} before it adds a
  package. Remove the overrides together with the path sources, or uv may pick any `isaaclab` release from the index.
- Isaac Lab extras requested through an `isaaclab[...]` requirement, by this project or by a registry package (for
  example `isaaclab[skrl]`), are not installed from the checkout. Use the dependency group of the same name, or add
  the extra to the `isaaclab-dev` group so that a plain `uv sync` keeps it installed, for example
  `isaaclab-dev = ["isaaclab-dev[rsl-rl,skrl]"]`.
- The checkout's packages report their own versions rather than {{ isaaclab_release }}, so `uv pip check` reports that
  this project requires `{{ isaaclab_requirement }}`. This is expected.
{% endif %}

## Run the tasks

{% if specifications %}
Replace the placeholders below with a task listed above and a selected RL library.

```bash
# List this project's environments and their available presets
uv run isaaclab list_envs --show_presets

# Exercise an environment without a trained policy
uv run isaaclab zero_agent --task <TASK_NAME> --num_envs 16
uv run isaaclab random_agent --task <TASK_NAME> --num_envs 16

# Train and play
uv run isaaclab train --rl_library <RL_LIBRARY> --task <TASK_NAME>
uv run isaaclab play --rl_library <RL_LIBRARY> --task <TASK_NAME> --checkpoint latest

# Distributed training
uv run isaaclab train_multigpu --rl_library <RL_LIBRARY> --task <TASK_NAME> --num_gpus 2

# Benchmark startup, runtime, training, or play
uv run isaaclab benchmark runtime --task <TASK_NAME> --num_envs 16 --num_steps 1000
uv run isaaclab benchmark training --rl_library <RL_LIBRARY> --task <TASK_NAME> --max_iterations 10
```

Use `physics=<PRESET>` to select one of the presets shown by `isaaclab list_envs`.
{% else %}
After registering a task, list it with:

```bash
uv run isaaclab list_envs --show_presets
```
{% endif %}

## Project structure

{% if specifications %}
Task families live in `src/{{ name }}/tasks`. Within a manager-based family, task-wide MDP terms live in `mdp`, while
robot-specific scenes, registrations, and agent configurations live in `config/{{ robot_name }}`. Add another robot by
creating a sibling of `config/{{ robot_name }}`; add another task by creating a sibling task-family directory.
{% else %}
Add task packages under `src/{{ name }}/tasks`. Import each task package from that directory so its Gymnasium
registrations load through the project's `isaaclab.tasks` entry point.
{% endif %}

## Project assets

Keep project-owned USD files and related data under `src/{{ name }}/assets/data`. The asset module exposes a
stable path that works from an editable checkout and an installed wheel:

```python
from {{ name }}.assets import {{ name | upper }}_ASSETS_DIR

robot_usd_path = {{ name | upper }}_ASSETS_DIR / "robots" / "my_robot.usd"
```

Keep asset configuration in the `{{ name }}.assets` package and pass `str(robot_usd_path)` to configuration fields that
expect a string. USD files are configured for Git LFS by the project's `.gitattributes`; install Git LFS before adding
large binary assets.

## Development

Run the registration test and code-quality checks through the project environment:

```bash
uv run pytest
uv run pre-commit run --all-files
```

The test helpers under `source/isaaclab_tasks/test` in the Isaac Lab repository are not part of the installed
`isaaclab_tasks` package. Keep test fixtures in this project and use public Isaac Lab APIs. If you copy
`env_test_utils.py`, it becomes vendored code whose upstream changes you must track.

To configure VS Code or Cursor, run the `setup_python_env` task or invoke its command directly:

```bash
uv run isaaclab --editor
```

The setup command selects the active interpreter and writes a git-ignored `pyrightconfig.json`. The resulting
configuration inherits the project's checked-in Pyright settings and adds the Isaac Sim extensions, project `src` root,
and any Isaac Lab packages discovered in the active Python environment. This supports both Pylance in VS Code and
basedpyright in Cursor.

In VS Code, use Pylance and select the interpreter that ran the setup command. In Cursor, install the
[basedpyright extension](https://marketplace.visualstudio.com/items?itemName=detachhead.basedpyright) instead of
Pylance, select the same interpreter, and reload the window. Both language servers read `pyrightconfig.json`.

When using the `isaacsim` {{ extra_kind }}, include it while generating the editor configuration so the command can
discover the Isaac Sim installation:

```bash
uv run {{ extra_option }} isaacsim isaaclab --editor
```

For an Isaac Sim binaries installation that is not available in the project environment, provide its path explicitly:

```bash
# Linux
uv run isaaclab --editor --isaac_path <isaac-sim-path>

# Windows
uv run isaaclab --editor --isaac_path <isaac-sim-path>
```

## Share it on the Isaac Lab registry

The Isaac Lab registry lists community task packages so that other users can find them with `isaaclab registry search`
and install them with `isaaclab registry install <name>`. The registry stores metadata and a pinned commit (or wheel
hash) only; your repository stays the source of truth.

1. Fill in `[tool.isaaclab.registry]` (category, tags, maintainers, and optional media) and `[project.urls] Repository`
   in `pyproject.toml`. Keep the published `isaaclab` requirement and its upper bound (`~={{ isaaclab_release }}`).
2. Commit your changes, tag the release with its version, and push both:

   ```bash
   git tag v0.1.0
   git push origin HEAD v0.1.0
   ```

3. Draft the registration request:

   ```bash
   uv run isaaclab registry publish --as <github-handle>
   ```

   `publish` reads `pyproject.toml` and the git state (the `origin` URL, falling back to `[project.urls] Repository`,
   the `v<version>` tag, and a clean tree). It
   runs pre-flight checks (an `isaaclab.tasks` entry point, a bounded `isaaclab` requirement, no `isaaclab-dev`
   requirement, a `LICENSE`, and a tag that points at `HEAD`) and writes the registration request. Use `--ref <tag>`
   for another tag, `--output <file>` to choose where the request is written, or `--pypi --index <url>` to register a
   wheel published on a package index instead of a git commit.

The registry bot validates the request, builds the wheel, and scans the registered tasks. A new package name is merged
after a waiting period; later versions from a listed maintainer are merged after a few minutes.

Packages installed into this project with `uv run isaaclab registry install <name>` are recorded in the
`isaaclab-registry` dependency group, which `uv sync` keeps installed. Remove one with
`uv run isaaclab registry uninstall <name>`. Inside this project, `uv run isaaclab list_envs` shows this project's tasks
and those of its registry packages; pass `--all` to include Isaac Lab's own tasks.

{% if include_ui_extension %}
## Isaac Sim UI extension

Add the project root to the Isaac Sim Extension Manager search paths, refresh, and enable the extension under
`Third Party`. Launch Isaac Sim through the project's `isaacsim` {{ extra_kind }} so the UI dependencies are
available. Kit loads `src/{{ name }}/ui_extension_example.py` through `config/extension.toml`.

{% endif %}
## Troubleshooting

If Pylance or basedpyright cannot resolve modules, confirm that the selected interpreter matches the one used to run the
setup command, then reload the editor window. To add a missing extension or reduce indexing memory, edit the `extraPaths`
array in the root `pyrightconfig.json`; remove simulator extension directories that the project does not use.
