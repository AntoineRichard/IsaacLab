Benchmarking
============

Isaac Lab ships three standalone benchmark scripts that emit a common
``v1.0`` JSON schema for training-performance and startup-performance data.
Downstream tooling (for example, the in-tree `Odin evaluation harness
<https://github.com/isaac-sim/IsaacLab/tree/main/tools/odin>`_) consumes these
JSON files, but the scripts are fully usable standalone.

.. contents::
   :local:
   :depth: 2


Scripts
-------

``benchmark_startup.py``
~~~~~~~~~~~~~~~~~~~~~~~~

Profiles five IsaacLab startup phases with ``cProfile``: ``app_launch``,
``python_imports``, ``task_config``, ``env_creation``, and ``first_step``. For
each phase it records wall-clock time and the top N self-time functions.

.. code-block:: bash

   ./isaaclab.sh -p scripts/benchmarks/benchmark_startup.py \
       --task Isaac-Ant-Direct-v0 --num_envs 4096 --headless \
       --schema_v1_output /tmp/startup.json

``benchmark_rsl_rl.py``
~~~~~~~~~~~~~~~~~~~~~~~

Trains a task with the RSL-RL PPO agent and records runtime / resource /
learning metrics, including exponentially-smoothed reward and episode-length
curves.

.. code-block:: bash

   ./isaaclab.sh -p scripts/benchmarks/benchmark_rsl_rl.py \
       --task Isaac-Ant-Direct-v0 --num_envs 4096 \
       --max_iterations 500 --headless \
       --schema_v1_output /tmp/training.json

``benchmark_skrl.py``
~~~~~~~~~~~~~~~~~~~~~

The SKRL-framework counterpart to ``benchmark_rsl_rl.py``. Emits the same
schema with ``framework: "skrl"``.

.. code-block:: bash

   ./isaaclab.sh -p scripts/benchmarks/benchmark_skrl.py \
       --task Isaac-Ant-Direct-v0 --num_envs 4096 \
       --max_iterations 500 --headless \
       --schema_v1_output /tmp/training_skrl.json


v1.0 schema summary
-------------------

Each script writes a single self-contained JSON file. The shape is defined by
dataclasses in :mod:`isaaclab.test.benchmark.standard_schema` — refer to the
module for per-field units and descriptions.

``TrainingBundle`` (training scripts) top-level keys:

* ``run`` — run identity (``run_id``, ``framework``, ``backend``, ``task``,
  ``seed``, ``num_envs``, ``max_iterations``, timestamps, ``status``).
* ``versions`` — software versions at run time (IsaacLab, Isaac Sim, Kit,
  Newton, Warp, Torch, RSL-RL / SKRL, git metadata).
* ``hardware`` — host snapshot (hostname, GPU devices, CPU, RAM).
* ``runtime`` — aggregated timings (``iterations_completed``,
  ``iteration_time_s``, ``env_steps_per_s``, ``iterations_per_s``,
  ``startup_phase_times_s``).
* ``resources`` — aggregated GPU/CPU/RAM utilisation (mean/std/peak).
* ``learning`` — final-value and EMA-smoothed reward / episode-length curves,
  with full per-iteration series unless ``--no_series`` is passed.

``StartupBundle`` (``benchmark_startup.py``) replaces ``runtime`` / ``resources``
/ ``learning`` with:

* ``phases`` — mapping from phase name to ``{total_time_s, top_functions}``.
* ``config`` — CLI configuration (``top_n``, ``whitelist``).


Common CLI flags
----------------

``--schema_v1_output <path>``
    Write the v1.0 JSON bundle to this path. If omitted, the script falls
    back to the legacy per-backend output format.

``--backend {physx, newton}``
    Physics backend tag recorded in the bundle. Passed by Odin wrappers; if
    omitted defaults to ``physx`` for the ``run.backend`` field.

``--run_id <string>``
    Explicit run-identity string. Odin wrappers compute this; if omitted a
    synthetic run_id of the form
    ``<framework>_<backend>_<task>_<YYYYMMDD-HHMMSS>_seed<seed>`` is generated.

``--ema_alpha <float>`` (training scripts)
    EMA smoothing factor for reward / episode-length curves (default
    ``0.05``, roughly a 20-sample window).

``--no_series`` (training scripts)
    Omit per-iteration series from the bundle, leaving only the
    ``final_raw`` + ``final_ema`` scalars.


Odin integration
----------------

The Odin evaluation harness under ``tools/odin/`` wraps these scripts to
produce multi-phase bundles (``startup.json`` + ``training.json`` + a thin
``manifest.json``). Standalone users can ignore Odin entirely; the v1.0 JSON
output is designed to be self-contained and directly loadable in dashboards.
