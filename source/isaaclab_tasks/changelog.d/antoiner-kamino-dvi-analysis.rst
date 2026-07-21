Added
^^^^^

* Added tuned Kamino DVI physics presets for Cartpole Direct and ANYmal-D Flat
  benchmarks.

Fixed
^^^^^

* Fixed ANYmal-D tuning analysis to retain exact failed screening preflights as
  auditable in-memory rejection evidence when the measured seed-42 run is absent.

* Fixed tuning artifact validation to accept the configured decision directory,
  derive it from staged output paths when omitted, and continue to reject
  undeclared artifact directories.
* Fixed the Kamino DVI tuning report to preserve and explain a terminal
  zero-survivor result without scheduling final or canonical runs or claiming
  a winner speedup.
