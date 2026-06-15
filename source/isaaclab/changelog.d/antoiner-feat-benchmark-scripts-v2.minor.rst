Added
^^^^^

* Added a shared, Isaac-Sim-free benchmark core under :mod:`isaaclab.test.benchmark`
  (``capture``, ``metrics``, ``builders``, ``stepping``, ``profiling``, and
  ``backend_descriptor`` submodules) that emits schema-v1 bundles, and unified
  benchmark entry points under ``scripts/benchmarks/``:

  - ``runtime.py`` — environment stepping without any RL library, emits a
    :class:`~isaaclab.test.benchmark.schema.RuntimeBundle`;
  - ``training.py`` — ``--rl_library`` dispatcher over rsl_rl / rl_games / skrl / sb3,
    emits a :class:`~isaaclab.test.benchmark.schema.TrainingBundle`;
  - ``startup.py`` — ``cProfile``-based startup-phase profiler, emits a
    :class:`~isaaclab.test.benchmark.schema.StartupBundle`.

  These replace the former per-backend ``benchmark_rsl_rl.py``,
  ``benchmark_rlgames.py``, ``benchmark_non_rl.py``, and ``benchmark_startup.py``
  scripts as well as the ``run_training_benchmarks.sh``,
  ``run_non_rl_benchmarks.sh``, and ``run_physx_benchmarks.sh`` CI shell runners.

Changed
^^^^^^^

* Routed the benchmark schema bundle through the :class:`~isaaclab.test.benchmark.BaseIsaacLabBenchmark`
  backend system as a new ``schema`` output format (now the default for ``runtime.py``/``training.py``/``startup.py``).
  ``--benchmark_backend`` accepts a comma-separated list (e.g. ``schema,omniperf``) to emit multiple formats in one run.
