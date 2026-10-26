
"""
Experiment 2: Effect of Basin Separation

Baseline algorithms:
    BO: Gaussian Process + Log Expected Improvement (BoTorch)
    CMA-ES: Standard CMA-ES without restarts

Objective:
    Imported from objective.py

Outputs:
    individual_runs.csv
    summary.csv
    convergence_raw.csv
    success_rate.png
    convergence.png
"""

import argparse
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import cma

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from objective import two_basin


# ============================================================
# 1. EXPERIMENT SETTINGS
# ============================================================

DISTANCES = [1, 2, 3, 4]

ALPHA = 10.0
DELTA = 1.0
DIM = 2

LOWER = -5.0
UPPER = 5.0

SUCCESS_TOL = 0.01
N_INITIAL = 8

NUM_RESTARTS = 5
RAW_SAMPLES = 64

DTYPE = torch.double
DEVICE = torch.device("cpu")


# ============================================================
# 2. OBJECTIVE AND PERFORMANCE METRICS
# ============================================================

def objective(x, distance):
    return two_basin(
        x,
        alpha=ALPHA,
        distance=distance,
        delta=DELTA
    )


def regret(value):
    # Known global optimum: -DELTA
    return max(0.0, float(value) + DELTA)


def unnormalise(x):
    return LOWER + (UPPER - LOWER) * x


# ============================================================
# 3. BAYESIAN OPTIMISATION: GP + LOGEI
# ============================================================

def run_bo(distance, budget, seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    acquisition_warnings = 0
    fallback_count = 0

    # Initial Sobol design
    sobol = torch.quasirandom.SobolEngine(
        dimension=DIM,
        scramble=True,
        seed=seed
    )

    n_init = min(N_INITIAL, budget)

    train_X = sobol.draw(n_init).to(
        dtype=DTYPE,
        device=DEVICE
    )

    initial_points = unnormalise(train_X).numpy()

    initial_values = [
        objective(x, distance)
        for x in initial_points
    ]

    train_Y = torch.tensor(
        initial_values,
        dtype=DTYPE,
        device=DEVICE
    ).unsqueeze(-1)

    best = float("inf")
    history = []

    for value in initial_values:
        best = min(best, value)
        history.append(regret(best))

    # Acquisition optimisation bounds: [0, 1]^2
    bounds = torch.stack([
        torch.zeros(DIM, dtype=DTYPE, device=DEVICE),
        torch.ones(DIM, dtype=DTYPE, device=DEVICE)
    ])

    while len(history) < budget:

        try:
            # Fit Gaussian Process
            model = SingleTaskGP(
                train_X,
                train_Y,
                outcome_transform=Standardize(m=1)
            )

            mll = ExactMarginalLogLikelihood(
                model.likelihood,
                model
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit_gpytorch_mll(mll)

            # Log Expected Improvement for minimisation
            acquisition = LogExpectedImprovement(
                model=model,
                best_f=train_Y.min().item(),
                maximize=False
            )

            # Track numerical optimisation warnings
            with warnings.catch_warnings(record=True) as caught:

                warnings.simplefilter(
                    "always", RuntimeWarning
                )

                candidate, _ = optimize_acqf(
                    acq_function=acquisition,
                    bounds=bounds,
                    q=1,
                    num_restarts=NUM_RESTARTS,
                    raw_samples=RAW_SAMPLES
                )

            warning_count = sum(
                issubclass(w.category, RuntimeWarning)
                for w in caught
            )

            acquisition_warnings += warning_count

            if warning_count > 0:
                print(
                    f"BO warning: distance={distance}, "
                    f"seed={seed}, "
                    f"eval={len(history)+1}, "
                    f"count={warning_count}"
                )

            candidate = candidate.detach()

            if not torch.isfinite(candidate).all():
                raise ValueError(
                    "Non-finite acquisition candidate"
                )

        except Exception as error:

            fallback_count += 1

            print(
                f"BO fallback: distance={distance}, "
                f"seed={seed}, "
                f"eval={len(history)+1}, "
                f"error={error}"
            )

            candidate = sobol.draw(1).to(
                dtype=DTYPE,
                device=DEVICE
            )

        # Evaluate the objective
        x_next = unnormalise(candidate[0]).numpy()
        y_next = objective(x_next, distance)

        # Update GP training data
        train_X = torch.cat(
            [train_X, candidate],
            dim=0
        )

        train_Y = torch.cat([
            train_Y,
            torch.tensor(
                [[y_next]],
                dtype=DTYPE,
                device=DEVICE
            )
        ], dim=0)

        best = min(best, y_next)
        history.append(regret(best))

    return {
        "algorithm": "BO-LogEI",
        "distance": distance,
        "seed": seed,
        "best_value": best,
        "final_regret": regret(best),
        "success": int(regret(best) <= SUCCESS_TOL),
        "evaluations_used": len(history),
        "acquisition_warnings": acquisition_warnings,
        "fallback_count": fallback_count,
        "history": history
    }


# ============================================================
# 4. STANDARD CMA-ES (NO RESTARTS)
# ============================================================

def run_cma(distance, budget, seed):

    np.random.seed(seed)

    initial_mean = [-2.0, 0.0]
    initial_sigma = 1.0

    options = {
        "seed": seed,
        "bounds": [LOWER, UPPER],
        "verbose": -9,
        "verb_disp": 0
    }

    es = cma.CMAEvolutionStrategy(
        initial_mean,
        initial_sigma,
        options
    )

    best = float("inf")
    history = []

    while len(history) < budget and not es.stop():

        solutions = es.ask()

        remaining = budget - len(history)
        n_evaluate = min(len(solutions), remaining)

        values = []

        for x in solutions[:n_evaluate]:

            y = objective(x, distance)
            values.append(y)

            best = min(best, y)
            history.append(regret(best))

        # Update only for a complete generation
        if n_evaluate == len(solutions):
            es.tell(solutions, values)
        else:
            break

    actual_evaluations = len(history)

    # Carry forward best-so-far regret if CMA-ES
    # terminates early. These are not new evaluations.
    if history:
        history += [history[-1]] * (budget - len(history))

    return {
        "algorithm": "CMA-ES",
        "distance": distance,
        "seed": seed,
        "best_value": best,
        "final_regret": regret(best),
        "success": int(regret(best) <= SUCCESS_TOL),
        "evaluations_used": actual_evaluations,
        "acquisition_warnings": 0,
        "fallback_count": 0,
        "history": history
    }


# ============================================================
# 5. RUN ALL EXPERIMENTS
# ============================================================

def run_experiment(runs, budget, output_dir):

    output_dir.mkdir(parents=True, exist_ok=True)

    individual_results = []
    convergence_results = []

    for distance in DISTANCES:

        print("\n" + "=" * 60)
        print(f"TESTING BASIN SEPARATION = {distance}")
        print("=" * 60)

        for seed_index in range(runs):

            seed = 1000 + seed_index

            for algorithm in [run_bo, run_cma]:

                result = algorithm(
                    distance=distance,
                    budget=budget,
                    seed=seed
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
                        "algorithm": result["algorithm"],
                        "seed": seed,
                        "evaluation": t,
                        "regret": r
                    })

                print(
                    f"{result['algorithm']:10s} "
                    f"seed={seed} "
                    f"regret={result['final_regret']:.6f} "
                    f"success={result['success']} "
                    f"evals={result['evaluations_used']} "
                    f"warnings={result['acquisition_warnings']} "
                    f"fallbacks={result['fallback_count']}"
                )

    # ========================================================
    # 6. SAVE INDIVIDUAL RESULTS
    # ========================================================

    individual_df = pd.DataFrame(individual_results)
    convergence_df = pd.DataFrame(convergence_results)

    individual_df.to_csv(
        output_dir / "individual_runs.csv",
        index=False
    )

    convergence_df.to_csv(
        output_dir / "convergence_raw.csv",
        index=False
    )

    # ========================================================
    # 7. SUMMARY STATISTICS
    # ========================================================

    summary = (
        individual_df
        .groupby(["distance", "algorithm"])
        .agg(
            success_rate=("success", "mean"),
            median_final_regret=("final_regret", "median"),
            mean_final_regret=("final_regret", "mean"),
            mean_evaluations_used=("evaluations_used", "mean"),
            total_acquisition_warnings=(
                "acquisition_warnings", "sum"
            ),
            total_fallbacks=("fallback_count", "sum")
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
    # 8. PLOT SUCCESS RATE
    # ========================================================

    plt.figure(figsize=(8, 5))

    for algorithm in summary["algorithm"].unique():

        data = summary[
            summary["algorithm"] == algorithm
        ]

        plt.plot(
            data["distance"],
            data["success_rate"] * 100,
            marker="o",
            linewidth=2,
            label=algorithm
        )

    plt.xlabel("Basin separation")
    plt.ylabel("Success rate (%)")
    plt.title("Effect of Basin Separation on Success")

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
    # 9. PLOT CONVERGENCE
    # ========================================================

    fig, axes = plt.subplots(
        2, 2,
        figsize=(12, 9),
        sharex=True
    )

    for ax, distance in zip(axes.flat, DISTANCES):

        data = convergence_df[
            convergence_df["distance"] == distance
        ]

        for algorithm in data["algorithm"].unique():

            subset = data[
                data["algorithm"] == algorithm
            ]

            median_curve = (
                subset
                .groupby("evaluation")["regret"]
                .median()
            )

            ax.plot(
                median_curve.index,
                np.maximum(median_curve.values, 1e-12),
                linewidth=2,
                label=algorithm
            )

        ax.set_title(f"Distance = {distance}")
        ax.set_xlabel("Objective evaluations")
        ax.set_ylabel("Median simple regret")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle(
        "BO-LogEI vs CMA-ES: Basin Separation"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / "convergence.png",
        dpi=300
    )
    plt.close()

    # ========================================================
    # 10. NUMERICAL DIAGNOSTICS
    # ========================================================

    bo_results = individual_df[
        individual_df["algorithm"] == "BO-LogEI"
    ]

    print("\nBO NUMERICAL DIAGNOSTICS")

    print(
        "Total acquisition warnings:",
        bo_results["acquisition_warnings"].sum()
    )

    print(
        "Total Sobol fallbacks:",
        bo_results["fallback_count"].sum()
    )

    print("\nResults saved to:", output_dir.resolve())


# ============================================================
# 11. MAIN
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
        default="results/experiment_2"
    )

    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")

    if args.budget < 1:
        parser.error("--budget must be at least 1")

    run_experiment(
        runs=args.runs,
        budget=args.budget,
        output_dir=Path(args.output)
    )
