"""Experiment 2 (layout sweep): effect of basin separation under three basin layouts.

Default: --layout fixed_midpoint --setup pilot. Each layout writes to results/experiment2_<layout>/.
Pass --output to use a _v2 folder (recommended if the old results/experiment2_<layout>/ folder
contains runs from the pre-margin runner).

  python code/experiment2_position.py --layout fixed_global   --setup pilot --workers 3
  python code/experiment2_position.py --layout fixed_midpoint --setup pilot --workers 3

--setup pilot reproduces the notebook CMA-ES (random start, sigma0 = 2) and notebook BO
(10 initial points, Normalize); use it for comparability with Experiment 2.
--setup aligned uses the layout-relative CMA-ES start (local basin centre, sigma0 = 1) and the
runner's BO with 8 initial points, for comparability with matched_init.py.
"""
import sys

import runner_position as runner

DEFAULTS = {
    "--cells": ["10:1", "10:2", "10:3", "10:4"],
    "--acquisitions": ["LogEI", "CMA-ES"],
    "--seed-start": [1000],
    "--runs": [30],
    "--budget": [300],
    "--setup": ["pilot"],
    "--layout": ["fixed_midpoint"],
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    if "--layout" in argv:
        i = argv.index("--layout")
        if i + 1 >= len(argv):
            raise SystemExit("--layout requires a value (original | fixed_global | fixed_midpoint)")
        layout = argv[i + 1]
    else:
        layout = DEFAULTS["--layout"][0]
    if layout not in runner.LAYOUTS:
        raise SystemExit(f"invalid --layout {layout!r}; expected one of {runner.LAYOUTS}")

    defaults = dict(DEFAULTS)
    defaults["--layout"] = [layout]
    if "--output" not in argv:
        defaults["--output"] = [f"results/experiment2_{layout}"]
    runner.run_with_defaults(defaults, argv)


if __name__ == "__main__":
    main()