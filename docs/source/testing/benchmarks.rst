.. _testing_benchmarks:

Benchmarking framework
======================

Isaac Lab provides a typed, Isaac-Sim-independent benchmark data model in
:mod:`isaaclab.test.benchmark` and unified scripts for measuring runtime,
startup, training, and policy inference. The scripts collect the same run,
version, hardware, timing, resource, and learning fields across supported RL
libraries and can write several output formats from one run.

Unified entry points
--------------------

Run benchmark scripts through the Isaac Lab Python environment:

* ``scripts/benchmarks/runtime.py`` steps an environment with random actions.
* ``scripts/benchmarks/startup.py`` profiles application launch, Python imports,
  task configuration, environment creation, and the first environment step.
* ``scripts/benchmarks/training.py`` dispatches training to RSL-RL, RL-Games,
  skrl, or Stable-Baselines3.
* ``scripts/benchmarks/play.py`` dispatches checkpoint inference to the same RL
  library adapters.

Every example below uses the Isaac Lab 2.x task name and explicitly selects the
supported PhysX preset.

Runtime
~~~~~~~

Measure environment stepping without an RL policy:

.. code-block:: bash

   ./isaaclab.sh -p scripts/benchmarks/runtime.py \
       --task Isaac-Cartpole-v0 \
       --num_envs 4096 \
       --num_frames 100 \
       --seed 42 \
       --benchmark_formatter schema,summary \
       --output_path ./benchmark_results \
       presets=physx --headless

The schema output is a :class:`~isaaclab.test.benchmark.RuntimeBundle` with
startup timing, per-step timing, throughput, hardware, versions, and resource
samples.

Startup
~~~~~~~

Profile the five startup phases independently with ``cProfile``:

.. code-block:: bash

   ./isaaclab.sh -p scripts/benchmarks/startup.py \
       --task Isaac-Cartpole-v0 \
       --num_envs 4096 \
       --top_n 30 \
       --benchmark_formatter schema,summary \
       --output_path ./benchmark_results \
       presets=physx --headless

Use ``--whitelist_config scripts/benchmarks/startup_whitelist.yaml`` to report
a stable set of function patterns for dashboards. A phase in the YAML uses its
whitelist; other phases retain ``--top_n`` behavior. Unmatched whitelist
patterns are emitted as zero-valued placeholders.

Training
~~~~~~~~

Select an adapter with ``--rl_library``. Valid values are ``rsl_rl``,
``rl_games``, ``skrl``, and ``sb3``.

.. code-block:: bash

   ./isaaclab.sh -p scripts/benchmarks/training.py \
       --rl_library rsl_rl \
       --task Isaac-Cartpole-v0 \
       --num_envs 4096 \
       --max_iterations 100 \
       --seed 42 \
       --benchmark_formatter schema,json \
       --output_path ./benchmark_results \
       presets=physx --headless

The training schema is a
:class:`~isaaclab.test.benchmark.TrainingBundle`. It adds learning curves,
convergence measurements, and success data when the task exposes it. Adapter
specific arguments remain available after ``--rl_library``; inspect a selected
adapter with ``--help`` for its exact options.

Training writes framework-native logs under ``logs/<rl-library>/`` as well as
formatter output under ``--output_path``. Each run directory contains a
``run.json`` manifest used to validate task, library, adapter metadata, and the
full Git SHA of both benchmark checkouts. When the comparison checkouts are not
available as sibling Git repositories, set ``ISAACLAB_BENCHMARK_LAB2_SHA`` and
``ISAACLAB_BENCHMARK_LAB3_SHA`` to their full 40-character SHAs.

Play
~~~~

Benchmark inference with an explicit checkpoint:

.. code-block:: bash

   ./isaaclab.sh -p scripts/benchmarks/play.py \
       --rl_library rsl_rl \
       --task Isaac-Cartpole-v0 \
       --num_envs 4096 \
       --num_frames 1000 \
       --checkpoint /path/to/model.pt \
       --benchmark_formatter schema,json \
       --output_path ./benchmark_results \
       presets=physx --headless

If ``--checkpoint`` is omitted, the adapter requests the published Isaac Lab
2.x checkpoint for the selected task. ``--checkpoint latest`` and
``--checkpoint best`` search compatible local training runs and reject
manifests whose task, library, adapter metadata, or comparison SHAs differ.
The resulting :class:`~isaaclab.test.benchmark.PlayBundle` records the resolved
checkpoint and inference throughput. Reward, episode length, and success rate
aggregate completed episodes only, so choose ``--num_frames`` long enough for
at least one episode to finish.

PhysX-only compatibility
------------------------

This Isaac Lab 2.x backport runs benchmarks with PhysX only. ``presets=physx``,
``physics=physx``, and the legacy default spelling are accepted compatibility
selectors. Presets for Newton, Kamino, MJWarp, or OV PhysX are Isaac Lab
3.0-only and fail before the simulator launches with guidance to use
``presets=physx``. Results should only be compared when task, seed, environment
count, frame or iteration count, hardware, and physics configuration match.

Output formatters and artifacts
-------------------------------

``--benchmark_formatter`` accepts one formatter or a comma-separated list:

.. list-table::
   :header-rows: 1
   :widths: 18 32 50

   * - Formatter
     - Output
     - Intended use
   * - ``schema``
     - One typed schema-v1 JSON bundle
     - Stable machine-readable runtime, startup, training, or play contract.
   * - ``json``
     - One JSON array of flat phases
     - Complete legacy measurements and metadata.
   * - ``osmo``
     - One JSON file per phase
     - Key-value KPI ingestion by Osmo.
   * - ``omniperf``
     - One phase-keyed JSON document
     - OmniPerf database upload and performance tracking.
   * - ``summary``
     - Human-readable stdout and one JSON file
     - Interactive inspection while retaining flat data.

Files use a timestamped workflow prefix. When several formatters are selected,
the formatter name is appended before ``.json``; Osmo then appends each phase
name. For example, a runtime run may produce
``benchmark_runtime_Isaac-Cartpole-v0_<timestamp>_schema.json`` and
``benchmark_runtime_Isaac-Cartpole-v0_<timestamp>_osmo_runtime.json``.

Formatter files are derived reports. Preserve the raw framework run directory
for reproducibility: it may contain TensorBoard event files, checkpoints,
configuration snapshots, videos, sensor captures, and ``run.json``. Checkpoint
and video fields in schema bundles refer to the corresponding raw artifacts;
generating a bundle does not copy them into ``--output_path``.

Legacy entry-point migration
----------------------------

The historical scripts remain available as deprecated compatibility wrappers.
They print migration guidance, translate supported workload arguments, and
delegate to the typed harness:

.. list-table::
   :header-rows: 1
   :widths: 34 40 26

   * - Deprecated command
     - Replacement
     - Added selector
   * - ``scripts/benchmarks/benchmark_non_rl.py``
     - ``scripts/benchmarks/runtime.py``
     - None
   * - ``scripts/benchmarks/benchmark_rlgames.py``
     - ``scripts/benchmarks/training.py``
     - ``--rl_library rl_games``
   * - ``scripts/benchmarks/benchmark_rsl_rl.py``
     - ``scripts/benchmarks/training.py``
     - ``--rl_library rsl_rl``

Replace the old ``--benchmark_backend`` value as follows:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Legacy backend
     - Unified formatter
   * - ``LocalLogMetrics``
     - ``--benchmark_formatter summary``
   * - ``JSONFileMetrics``
     - ``--benchmark_formatter json``
   * - ``OsmoKPIFile``
     - ``--benchmark_formatter osmo``
   * - ``OmniPerfKPIFile``
     - ``--benchmark_formatter omniperf``

The legacy default remains OmniPerf. New automation should invoke the unified
entry points directly so it can select ``schema`` and multiple formatters. The
legacy ``--video``, ``--video_length``, ``--video_interval``, and
``--distributed`` modes are outside this migration path and are not supported
by the unified entry points. Remove those flags when migrating a benchmark
command; capture benchmark artifacts separately when needed.

Python API
----------

Use the public functions in :mod:`isaaclab.test.benchmark.builders` to assemble
a typed bundle and :func:`~isaaclab.test.benchmark.write_bundle_file` to write
the completed schema atomically. The unified scripts combine those functions
with :class:`~isaaclab.test.benchmark.BaseIsaacLabBenchmark` and
:class:`~isaaclab.test.benchmark.BenchmarkMonitor` for recorder-backed resource
sampling.

.. code-block:: python

   from isaaclab.test.benchmark import builders, write_bundle_file

   runtime = builders.build_runtime(
       startup_time_s=startup_time,
       iteration_times_s=iteration_times,
       collection_fps=collection_fps,
       total_fps=total_fps,
       steps_per_iteration=num_envs,
   )
   bundle = builders.build_runtime_bundle(
       run=run_identity,
       versions=versions,
       hardware=hardware,
       runtime=runtime,
       resources=resources,
   )
   write_bundle_file(bundle, "./benchmark_results/runtime.json")

See :mod:`isaaclab.test.benchmark` for measurement, schema, builder, capture,
serialization, profiling, and stepping utilities.
