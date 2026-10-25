import json
import numpy as np
from landscape import TwoBasinGaussian
from runners import run_cmaes, run_bo

WIDTHS = [1.0, 0.25]
SEEDS = [0, 1, 2]
BUDGET = 300

results = []
for sigma_g in WIDTHS:
    L = TwoBasinGaussian(d=5, sigma_g=sigma_g)
    for seed in SEEDS:
        for runner in (run_cmaes, run_bo):
            r = runner(L, budget=BUDGET, seed=seed)
            r["sigma_g"] = sigma_g
            r["seed"] = seed
            # drop heavy arrays for printing
            summary = {k: v for k, v in r.items() if k not in ("X", "f", "basins")}
            results.append(summary)
            print(json.dumps(summary))

with open("../results/pilot.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to results/pilot.json")
