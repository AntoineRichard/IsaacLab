# Native DLSS capture selection evidence

Visual evidence for https://github.com/isaac-sim/IsaacLab/pull/8882 at commit `aeead71e1be6b3715b4f4247ec835caa96dbfd5a`.

Figures use native OVRTX 0.6.0 / OVStage 0.3.0 captures from an RTX 5090, driver 610.57.04, at 1280x720 and 60 Hz. The reproduction uses an LDR-only product and companion settings described in the PR. No images were regenerated.

Before selects the first returned intermediate image. After selects the latest real capture. The two saved real captures exactly match the generation-disabled control. The error map shows maximum absolute RGB channel difference per pixel on a fixed 0-80 scale. Blue boxes and lower-row zooms use identical bounds, with nearest-neighbor display.

Frame 2: capture times 41.6666665 vs 50 ms; 129,036 differing RGB channel values, maximum error 78/255. Frame 15: capture times 258.3333333 vs 266.6666667 ms; 87,969 differing channel values, maximum error 79/255. Fixed-versus-control error is zero.
