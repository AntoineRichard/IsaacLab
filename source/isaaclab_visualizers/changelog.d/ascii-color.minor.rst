Added
^^^^^

* Added ``AsciiVisualizerCfg.color``, which draws the terminal view in color with one color per
  body instead of shading characters. It rasterizes onto quadrant block characters, which carry
  2x2 subpixels and two colors per cell, so it resolves four times the detail. It needs a
  terminal with 24-bit color, so shading characters remain the default.
