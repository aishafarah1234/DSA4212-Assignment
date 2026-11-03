"""Experiment 3: effect of the BO acquisition function.

LogEI (baseline), LogPI, UCB (beta = 0.5, 4) plus Random and Sobol sampling on four landscapes selected from
Stage 1: alpha 5, 10, 20 at distance 4 and distance 1 at alpha 10; 200 evaluations, 30 fresh seeds (2000-2029).
The UCB diagnostic is the same engine with --cells 5:4 --acquisitions LogEI UCB-4 UCB-64 UCB-256 --runs 10
--output results/exp3_diagnostic.

  python code/experiment3.py --workers 3

Uses the shared engine in runner.py; only the defaults below are specific to this experiment. Any option can be
overridden (e.g. --runs 2 --budget 30 --output results/smoke for a smoke test; --aggregate-only to rebuild tables).
Results go to results/experiment3/ (runs/*.json, summary.csv, paired_tests_vs_logei.csv, figures).
"""
import runner

DEFAULTS = {
    "--cells": ["5:4", "10:4", "20:4", "10:1"],
    "--acquisitions": [
        "LogEI",
        "LogPI",
        "UCB-0.5",
        "UCB-4",
        "Random",
        "Sobol"
    ],
    "--seed-start": [2000],
    "--runs": [30],
    "--budget": [300],
    "--setup": ["pilot"],
    "--output": ["results/experiment3"],
}

def main(argv=None):
    runner.run_with_defaults(DEFAULTS, argv)

if __name__ == "__main__":
    main()
