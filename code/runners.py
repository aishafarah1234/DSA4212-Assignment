
import numpy as np
import cma
import torch

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize

from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import qmc

def run_cmaes(landscape, budget=300, seed=0):
    rng = np.random.default_rng(seed)

    d = landscape.d
    lower = landscape.bounds[:, 0]
    upper = landscape.bounds[:, 1]

    x0 = rng.uniform(lower, upper, size=d)
    sigma0 = 3.0

    es = cma.CMAEvolutionStrategy(
        x0,
        sigma0,
        {
            "bounds": [lower.tolist(), upper.tolist()],
            "maxfevals": budget,
            "verbose": -9,
            "seed": seed,
        }
    )

    X_hist = []
    f_hist = []
    n_evals = 0

    while not es.stop() and n_evals < budget:
        X = np.asarray(es.ask(), dtype=float)
        X = np.clip(X, lower, upper)

        remaining = budget - n_evals

        # CMA-ES requires a complete generation for es.tell().
        # Stop before exceeding the evaluation budget.
        if len(X) > remaining:
            break

        f = np.asarray(landscape(X), dtype=float).reshape(-1)

        es.tell(X.tolist(), f.tolist())

        X_hist.extend(X.tolist())
        f_hist.extend(f.tolist())

        n_evals += len(X)

    X_hist = np.asarray(X_hist)
    f_hist = np.asarray(f_hist)

    if len(f_hist) == 0:
        raise RuntimeError("CMA-ES produced no evaluations.")

    basins = landscape.basin_of(X_hist)
    best_idx = int(np.argmin(f_hist))

    return {
        "method": "cmaes",
        "best_final": float(f_hist[best_idx]),
        "best_point": X_hist[best_idx].tolist(),
        "success": int(basins[best_idx] == 0),
        "first_entry_index": (
            int(np.argmax(basins == 0))
            if (basins == 0).any()
            else None
        ),
        "frac_global": float((basins == 0).mean()),
        "frac_local": float((basins == 1).mean()),
    }

def run_bo(
    landscape,
    budget=300,
    seed=0,
    n_init=None,
    gp_maxiter=50,
    verbose=False,
):
    torch.manual_seed(seed)

    d = landscape.d
    n_init = 2 * d + 5 if n_init is None else n_init

    if budget < 1:
        raise ValueError("budget must be at least 1.")

    n_init = min(n_init, budget)

    dtype = torch.double
    device = torch.device("cpu")

    # Original search bounds: shape (2, d)
    bounds = torch.tensor(
        landscape.bounds.T,
        dtype=dtype,
        device=device,
    )

    # Normalized search bounds: [0, 1]^d
    normalized_bounds = torch.stack([
        torch.zeros(d, dtype=dtype, device=device),
        torch.ones(d, dtype=dtype, device=device),
    ])

    sampler = qmc.LatinHypercube(d=d, seed=seed)

    X_init_normalized = torch.tensor(
        sampler.random(n_init),
        dtype=dtype,
        device=device,
    )

    X_init = unnormalize(X_init_normalized, bounds)

    Y_init = torch.tensor(
        np.asarray(landscape(X_init.numpy())).reshape(-1, 1),
        dtype=dtype,
        device=device,
    )

    X_all = X_init.clone()
    Y_all = Y_init.clone()

    while X_all.shape[0] < budget:

        iteration = X_all.shape[0]

        if verbose:
            print(
                f"BO evaluation {iteration + 1}/{budget} "
                f"| Best = {Y_all.min().item():.6f}",
                flush=True,
            )

        # Normalize all observed input locations
        X_normalized = normalize(X_all, bounds)

        gp = SingleTaskGP(
            train_X=X_normalized,
            train_Y=Y_all,
        )

        mll = ExactMarginalLogLikelihood(
            gp.likelihood,
            gp,
        )

        fit_gpytorch_mll(
            mll,
            optimizer_kwargs={
                "options": {
                    "maxiter": gp_maxiter,
                }
            },
        )

        log_ei = LogExpectedImprovement(
            model=gp,
            best_f=Y_all.min().item(),
            maximize=False,
        )

        candidate_normalized, _ = optimize_acqf(
            acq_function=log_ei,
            bounds=normalized_bounds,
            q=1,
            num_restarts=5,
            raw_samples=64,
        )

        # Convert back to original search coordinates
        candidate = unnormalize(
            candidate_normalized,
            bounds,
        )

        y_new = torch.tensor(
            np.asarray(landscape(candidate.detach().numpy()))
            .reshape(-1, 1),
            dtype=dtype,
            device=device,
        )

        # Update observations
        X_all = torch.cat(
            [X_all, candidate.detach()],
            dim=0,
        )

        Y_all = torch.cat(
            [Y_all, y_new],
            dim=0,
        )
        
    X_np = X_all.detach().numpy()
    f_np = Y_all.squeeze(-1).detach().numpy()

    basins = landscape.basin_of(X_np)
    best_idx = int(np.argmin(f_np))

    if verbose:
        print(
            f"BO completed | Best = {f_np[best_idx]:.6f}",
            flush=True,
        )

    return {
        "method": "bo",
        "best_final": float(f_np[best_idx]),
        "best_point": X_np[best_idx].tolist(),
        "success": int(basins[best_idx] == 0),
        "first_entry_index": (
            int(np.argmax(basins == 0))
            if (basins == 0).any()
            else None
        ),
        "frac_global": float((basins == 0).mean()),
        "frac_local": float((basins == 1).mean()),
    }
