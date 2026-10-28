
"""
Experiment 4: Effect of CMA-ES Restart Strategies

Compare:
1. Standard CMA-ES (no restarts)
2. Random-restart CMA-ES
3. IPOP-CMA-ES (increasing population)

All configurations:
- Use the same initial mean and sigma
- Have a maximum of 200 objective evaluations
- Use identical objective functions
- Record actual restarts and evaluations

Outputs:
- individual_runs.csv
- summary.csv
- convergence_raw.csv
- success_rate.png
- convergence.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cma

from objective import two_basin


# ============================================================
# 1. EXPERIMENT SETTINGS
# ============================================================

DISTANCES = [1, 4]

ALPHA = 10.0
DELTA = 1.0

DIM = 2
LOWER = -5.0
UPPER = 5.0

SUCCESS_TOL = 0.01

INITIAL_MEAN = np.array([-2.0, 0.0])
INITIAL_SIGMA = 1.0

# Default CMA-ES population size for 2 dimensions
POP_SIZE = 4 + int(3 * np.log(DIM))

# Maximum evaluations allocated to each restart segment
RESTART_INTERVAL = 60

# IPOP population multiplier
IPOP_FACTOR = 2

METHODS = [
    "CMA-ES",
    "CMA-Restart",
    "CMA-IPOP"
]


# ============================================================
# 2. OBJECTIVE AND METRICS
# ============================================================

def objective(x, distance):
    return two_basin(
        x,
        alpha=ALPHA,
        distance=distance,
        delta=DELTA
    )


def regret(value):
    # Global optimum is -DELTA
    return max(0.0, float(value) + DELTA)


# ============================================================
# 3. CMA-ES RUNNER
# ============================================================

def run_cma(distance, budget, seed, method, restart_interval=60):

    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")

    rng = np.random.default_rng(seed)

    best = float("inf")
    history = []

    restart_count = 0
    current_popsize = POP_SIZE

    # Same initial seed for all methods.
    # Subsequent restarts receive new seeds.
    first_run = True

    while len(history) < budget:

        if first_run:
            mean = INITIAL_MEAN.copy()
            cma_seed = seed
        else:
            mean = rng.uniform(
                LOWER, UPPER, size=DIM
            )
            cma_seed = int(
                rng.integers(1, 2**31 - 1)
            )

        options = {
            "seed": cma_seed,
            "bounds": [LOWER, UPPER],
            "popsize": current_popsize,
            "verbose": -9,
            "verb_disp": 0
        }

        es = cma.CMAEvolutionStrategy(
            mean,
            INITIAL_SIGMA,
            options
        )

        evaluations_this_segment = 0

        # Baseline: no scheduled restarts
        if method == "CMA-ES":
            segment_limit = budget
        else:
            segment_limit = restart_interval

        # ====================================================
        # CMA-ES GENERATIONS
        # ====================================================

        while (
            len(history) < budget
            and evaluations_this_segment < segment_limit
            and not es.stop()
        ):

            solutions = es.ask()

            remaining_budget = budget - len(history)

            remaining_segment = (
                segment_limit - evaluations_this_segment
            )

            n_evaluate = min(
                len(solutions),
                remaining_budget,
                remaining_segment
            )

            values = []

            for x in solutions[:n_evaluate]:

                y = objective(x, distance)

                values.append(y)

                best = min(best, y)
                history.append(regret(best))

                evaluations_this_segment += 1

            # Update CMA-ES only for full generations
            if n_evaluate == len(solutions):
                es.tell(solutions, values)
            else:
                break

        first_run = False

        # Baseline stops without restarting
        if method == "CMA-ES":
            break

        # No restart when budget is exhausted
        if len(history) >= budget:
            break

        # Start a new CMA-ES run
        restart_count += 1

        # IPOP doubles population at each restart
        if method == "CMA-IPOP":
            current_popsize = max(
                current_popsize + 1,
                int(current_popsize * IPOP_FACTOR)
            )

    actual_evaluations = len(history)

    # Carry forward regret if the baseline stops early.
    # Padding does NOT represent extra evaluations.
    if actual_evaluations < budget:
        history += [history[-1]] * (
            budget - actual_evaluations
        )

    return {
    "distance": distance,
    "algorithm": method,
    "seed": seed,
    "best_value": best,
    "final_regret": regret(best),

    # Strict convergence success
    "success": int(regret(best) <= SUCCESS_TOL),

    # Global basin discovery
    "basin_discovered": int(best < 0.0),

    "evaluations_used": actual_evaluations,
    "restart_count": restart_count,
    "history": history
    }


# ============================================================
# 4. RUN EXPERIMENT
# ============================================================

def run_experiment(runs, budget, output_dir, restart_interval):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    individual_results = []
    convergence_results = []

    for distance in DISTANCES:

        print("\n" + "=" * 60)
        print(f"BASIN SEPARATION = {distance}")
        print("=" * 60)

        for seed_index in range(runs):

            seed = 1000 + seed_index

            for method in METHODS:

                result = run_cma(
                    distance=distance,
                    budget=budget,
                    seed=seed,
                    method=method,
                    restart_interval=restart_interval
                )

                individual_results.append({
                    k: v
                    for k, v in result.items()
                    if k != "history"
                })

                for t, r in enumerate(
                    result["history"],
                    start=1
                ):
                    convergence_results.append({
                        "distance": distance,
                        "algorithm": method,
                        "seed": seed,
                        "evaluation": t,
                        "regret": r
                    })

                print(
                    f"{method:13s} "
                    f"seed={seed} "
                    f"regret={result['final_regret']:.6f} "
                    f"success={result['success']} "
                    f"evals={result['evaluations_used']} "
                    f"restarts={result['restart_count']}"
                )

    # ========================================================
    # 5. SAVE RAW RESULTS
    # ========================================================

    individual_df = pd.DataFrame(
        individual_results
    )

    convergence_df = pd.DataFrame(
        convergence_results
    )

    individual_df.to_csv(
        output_dir / "individual_runs.csv",
        index=False
    )

    convergence_df.to_csv(
        output_dir / "convergence_raw.csv",
        index=False
    )

    # ========================================================
    # 6. SUMMARY STATISTICS
    # ========================================================

    summary = (
        individual_df
        .groupby(["distance", "algorithm"])
        .agg(
            success_rate=("success", "mean"),
            discovery_rate=("basin_discovered", "mean"),
            median_final_regret=(
                "final_regret", "median"
            ),
            mean_final_regret=(
                "final_regret", "mean"
            ),
            mean_evaluations_used=(
                "evaluations_used", "mean"
            ),
            mean_restart_count=(
                "restart_count", "mean"
            )
        )
        .reset_index()
    )

    summary.to_csv(
        output_dir / "summary.csv",
        index=False
    )

    print("\nEXPERIMENT SUMMARY")
    print(summary.to_string(index=False))

    # ========================================================
    # 7. SUCCESS RATE PLOT
    # ========================================================

    plt.figure(figsize=(9, 5))

    for method in METHODS:

        data = summary[
            summary["algorithm"] == method
        ]

        plt.plot(
            data["distance"],
            data["success_rate"] * 100,
            marker="o",
            linewidth=2,
            label=method
        )

    plt.xlabel("Basin separation")
    plt.ylabel("Success rate (%)")

    plt.title(
        "Effect of CMA-ES Restart Strategies"
    )

    plt.xticks(DISTANCES)
    plt.ylim(-5, 105)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "success_rate.png",
        dpi=300
    )

    plt.close()

    # ========================================================
    # 8. CONVERGENCE PLOT
    # ========================================================

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 5),
        sharey=True
    )

    for ax, distance in zip(axes, DISTANCES):

        data = convergence_df[
            convergence_df["distance"] == distance
        ]

        for method in METHODS:

            subset = data[
                data["algorithm"] == method
            ]

            median_curve = (
                subset
                .groupby("evaluation")["regret"]
                .median()
            )

            ax.plot(
                median_curve.index,
                np.maximum(
                    median_curve.values,
                    1e-12
                ),
                linewidth=2,
                label=method
            )

        ax.set_title(f"Distance = {distance}")
        ax.set_xlabel("Objective evaluations")
        ax.set_ylabel("Median simple regret")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle(
        "CMA-ES Restart Strategy Comparison"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / "convergence.png",
        dpi=300
    )

    plt.close()

    print(
        "\nResults saved to:",
        output_dir.resolve()
    )


# ============================================================
# 9. MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--runs",
        type=int,
        default=30
    )

    parser.add_argument(
        "--budget",
        type=int,
        default=200
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/experiment4_separation"
    )

    parser.add_argument(
        "--restart-interval",
        type=int,
        default=60,
        help="Maximum evaluations per restart segment"
    )

    # Parse arguments AFTER defining all arguments
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be positive")

    if args.budget < 1:
        parser.error("--budget must be positive")

    if args.restart_interval < 1:
        parser.error("--restart-interval must be positive")

    # Run the experiment only once
    run_experiment(
        runs=args.runs,
        budget=args.budget,
        output_dir=Path(args.output),
        restart_interval=args.restart_interval
    )
