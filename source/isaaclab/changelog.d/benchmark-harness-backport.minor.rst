Added
^^^^^

* Added unified runtime, startup, training, and play benchmark entry points with
  typed schema output and JSON, Osmo, OmniPerf, and summary formatters.

Deprecated
^^^^^^^^^^

* Deprecated ``scripts/benchmarks/benchmark_non_rl.py`` in favor of
  ``scripts/benchmarks/runtime.py``. Migrate ``--benchmark_backend`` to the
  corresponding ``--benchmark_formatter`` value.
* Deprecated ``scripts/benchmarks/benchmark_rlgames.py`` in favor of
  ``scripts/benchmarks/training.py --rl_library rl_games``. Migrate
  ``--benchmark_backend`` to the corresponding ``--benchmark_formatter`` value.
* Deprecated ``scripts/benchmarks/benchmark_rsl_rl.py`` in favor of
  ``scripts/benchmarks/training.py --rl_library rsl_rl``. Migrate
  ``--benchmark_backend`` to the corresponding ``--benchmark_formatter`` value.
* Deprecated legacy video and distributed benchmark modes without replacement.
  Remove ``--video``, ``--video_length``, ``--video_interval``, and
  ``--distributed`` when migrating to the unified benchmark entry points.
