# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Watch a scripted Franka FR3 fill a Christmas stocking in Isaac Lab.

.. code-block:: bash

    uv run isaaclab demo christmas-atelier
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from christmas_atelier.run import main  # noqa: E402

if __name__ == "__main__":
    main()
