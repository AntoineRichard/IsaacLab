Fixed
^^^^^

* Fixed :attr:`~isaaclab.benchmark.schema.TrainingBundle.checkpoint_path` being reported as
  ``None`` by the RL-Games and SKRL training benchmarks. The field is now populated from the
  checkpoint the run actually wrote, so a chained play step can roll out a freshly trained
  policy without reconstructing the path. The search matches every library's naming,
  including SKRL's ``agent_<tag>.pt`` and ``best_agent.pt``.
