"""Experiment 3: BO acquisition choices on nearby and distant two-basin landscapes.

Uses the same baseline as Experiment 2: SingleTaskGP on f, Standardize,
8 scrambled Sobol initial points, LogEI with maximize=False, 5 restarts,
64 raw samples, and 200 total evaluations by default.
"""
import argparse
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import LogExpectedImprovement, LogProbabilityOfImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from objective import two_basin

DISTANCES = (1, 4)
ALPHA, DELTA = 10.0, 1.0
DIM, LOWER, UPPER = 2, -5.0, 5.0
SUCCESS_TOL, N_INITIAL = 0.01, 8
NUM_RESTARTS, RAW_SAMPLES = 5, 64
DTYPE = torch.double
ACQUISITIONS = ('LogEI', 'LogPI', 'UCB-0.5', 'UCB-4')


def objective(x, distance):
    return two_basin(x, alpha=ALPHA, distance=distance, delta=DELTA)


def regret(value):
    return max(0.0, float(value) + DELTA)


def unnormalise(u):
    return LOWER + (UPPER - LOWER) * u


def run_bo(distance, budget, seed, acquisition_name):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    sobol = torch.quasirandom.SobolEngine(DIM, scramble=True, seed=seed)
    train_X = sobol.draw(min(N_INITIAL, budget)).to(dtype=DTYPE)
    initial_values = [objective(unnormalise(u).numpy(), distance) for u in train_X]
    train_Y = torch.tensor(initial_values, dtype=DTYPE).unsqueeze(-1)
    best = float('inf')
    history = []
    for y in initial_values:
        best = min(best, y)
        history.append(regret(best))

    bounds = torch.stack([torch.zeros(DIM, dtype=DTYPE), torch.ones(DIM, dtype=DTYPE)])
    acquisition_warnings = 0
    fallback_count = 0
    while len(history) < budget:
        try:
            # Fit the same GP to the ORIGINAL objective for every acquisition.
            model = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                fit_gpytorch_mll(mll)

            if acquisition_name == 'LogEI':
                acq = LogExpectedImprovement(model=model, best_f=train_Y.min().item(), maximize=False)
            elif acquisition_name == 'LogPI':
                acq = LogProbabilityOfImprovement(model=model, best_f=train_Y.min().item(), maximize=False)
            elif acquisition_name.startswith('UCB-'):
                # UCB maximises; apply it to -f via a posterior mean transform.
                from botorch.acquisition.objective import ScalarizedPosteriorTransform
                negate = ScalarizedPosteriorTransform(weights=torch.tensor([-1.0], dtype=DTYPE))
                acq = UpperConfidenceBound(model=model, beta=float(acquisition_name.split('-')[1]), posterior_transform=negate)
            else:
                raise ValueError(f'Unknown acquisition: {acquisition_name}')

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', RuntimeWarning)
                candidate, _ = optimize_acqf(acq_function=acq, bounds=bounds, q=1,
                                              num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES)
            acquisition_warnings += sum(issubclass(w.category, RuntimeWarning) for w in caught)
            candidate = candidate.detach()
            if not torch.isfinite(candidate).all():
                raise ValueError('Non-finite acquisition candidate')
        except Exception as exc:
            fallback_count += 1
            print(f'Fallback: distance={distance} seed={seed} acquisition={acquisition_name} '
                  f'evaluation={len(history)+1}: {exc}')
            candidate = sobol.draw(1).to(dtype=DTYPE)

        y = objective(unnormalise(candidate[0]).numpy(), distance)
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, torch.tensor([[y]], dtype=DTYPE)], dim=0)
        best = min(best, y)
        history.append(regret(best))

    return dict(distance=distance, algorithm=f'BO-{acquisition_name}', seed=seed,
                best_value=best, final_regret=regret(best),
                success=int(regret(best) <= SUCCESS_TOL), evaluations_used=len(history),
                acquisition_warnings=acquisition_warnings, fallback_count=fallback_count,
                history=history)


def run_experiment(runs, budget, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, curves = [], []
    for distance in DISTANCES:
        for seed_index in range(runs):
            seed = 1000 + seed_index
            for acq in ACQUISITIONS:
                result = run_bo(distance, budget, seed, acq)
                rows.append({k: v for k, v in result.items() if k != 'history'})
                curves.extend(dict(distance=distance, algorithm=result['algorithm'], seed=seed,
                                   evaluation=i, regret=float(r))
                              for i, r in enumerate(result['history'], start=1))
                print(f"distance={distance} {result['algorithm']} seed={seed} "
                      f"regret={result['final_regret']:.6g} success={result['success']} "
                      f"warnings={result['acquisition_warnings']} fallbacks={result['fallback_count']}")
    individual = pd.DataFrame(rows)
    convergence = pd.DataFrame(curves)
    individual.to_csv(output_dir / 'individual_runs.csv', index=False)
    convergence.to_csv(output_dir / 'convergence_raw.csv', index=False)
    summary = individual.groupby(['distance', 'algorithm'], as_index=False).agg(
        success_rate=('success', 'mean'),
        median_final_regret=('final_regret', 'median'),
        mean_final_regret=('final_regret', 'mean'),
        mean_evaluations_used=('evaluations_used', 'mean'),
        total_acquisition_warnings=('acquisition_warnings', 'sum'),
        total_fallbacks=('fallback_count', 'sum'))
    summary.to_csv(output_dir / 'summary.csv', index=False)
    print('\nEXPERIMENT 3 SUMMARY\n', summary.to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(ACQUISITIONS))
    width = 0.35
    for j, distance in enumerate(DISTANCES):
        data = summary.set_index(['distance', 'algorithm'])
        rates = [100 * data.loc[(distance, f'BO-{a}'), 'success_rate'] for a in ACQUISITIONS]
        ax.bar(x + (j - 0.5) * width, rates, width, label=f'Distance {distance}')
    ax.set_xticks(x, [f'BO-{a}' for a in ACQUISITIONS])
    ax.set_ylabel('Success rate (%)')
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'success_rate.png', dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(DISTANCES), figsize=(12, 5), sharey=True)
    for ax, distance in zip(axes, DISTANCES):
        data = convergence[convergence['distance'] == distance]
        for algorithm in [f'BO-{a}' for a in ACQUISITIONS]:
            curve = data[data['algorithm'] == algorithm].groupby('evaluation')['regret'].median()
            ax.plot(curve.index, np.maximum(curve.values, 1e-12), label=algorithm)
        ax.set_yscale('log')
        ax.set_title(f'Distance = {distance}')
        ax.set_xlabel('Objective evaluations')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('Median simple regret')
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / 'convergence.png', dpi=300)
    plt.close(fig)
    print(f'\nSaved results to {output_dir.resolve()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=30)
    parser.add_argument('--budget', type=int, default=200)
    parser.add_argument('--output', type=Path, default=Path('results/experiment3_separation'))
    args = parser.parse_args()
    if args.runs < 1 or args.budget < 1:
        parser.error('--runs and --budget must be positive')
    run_experiment(args.runs, args.budget, args.output)
