* Added the experimental ``isaaclab registry`` command to list, search, inspect, install, uninstall, and restore
  community task packages from a registry index, and to draft registration requests with ``publish``. Inside a
  user's uv project, packages are added to its ``isaaclab-registry`` dependency group with ``uv add``; elsewhere
  they are installed into the running environment with ``uv pip install``, keeping the installed Isaac Lab in place
  and rolling back an install that introduces dependency conflicts.
