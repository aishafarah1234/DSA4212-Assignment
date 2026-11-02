"""Experiment 2: effect of the basin separation.

BO (LogEI) and default CMA-ES on distance in {1, 2, 3, 4} at alpha = 10, 200 evaluations, 30 seeds (1000-1029).
"LogEI" in the tables is BO.

  python code/experiment2.py --workers 3

Uses the shared engine in runner.py; only the defaults below are specific to this experiment. Any option can be
overridden (e.g. --runs 2 --budget 30 --output results/smoke for a smoke test; --aggregate-only to rebuild tables).
Results go to results/experiment2/ (runs/*.json, summary.csv, paired_tests_vs_logei.csv, figures).
"""
import runner

DEFAULTS = {
    "--cells": ["10:1", "10:2", "10:3", "10:4"],
    "--acquisitions": ["LogEI", "CMA-ES"],
    "--seed-start": [1000],
    "--runs": [30],
    "--budget": [300],
    "--setup": ["pilot"],
    "--output": ["results/experiment2"],
}

def main(argv=None):
    runner.run_with_defaults(DEFAULTS, argv)

if __name__ == "__main__":
    main()
