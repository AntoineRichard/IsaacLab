* Added the experimental ``isaaclab registry`` command to list, search, inspect, install, and uninstall community
  task packages from a registry index, to show how the installed task packages relate to it, and to check a tagged
  project and draft its registration request with ``publish``. Inside a user's uv project, packages are added to its
  ``isaaclab-registry`` dependency group with ``uv add``; elsewhere they are installed into the running environment
  with ``uv pip install``. Releases are pinned to a git commit or a wheel sha256, and the installed Isaac Lab is
  never replaced.
