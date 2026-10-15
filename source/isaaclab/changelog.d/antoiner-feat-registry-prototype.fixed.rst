* Fixed ``isaaclab --new`` silently leaving a repository without commits when ``git commit`` fails (for example
  without a configured git identity); it now prints a warning with the command to run.
