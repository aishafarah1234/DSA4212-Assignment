import numpy as np
import cma
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from scipy.stats import qmc


def run_cmaes(landscape, budget=300, seed=0):
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(-5, 5, size=landscape.d)
    sigma0 = 3.0
    es = cma.CMAEvolutionStrategy(
        x0, sigma0,
        {'bounds': [list(landscape.bounds[:, 0]), list(landscape.bounds[:, 1])],
         'maxfevals': budget,
         'verbose': -9,
         'seed': seed,
         'CMA_stds': None}
    )
    n_evals = 0
    X_hist, f_hist = [], []
    while not es.stop() and n_evals < budget:
        X = es.ask()
        # clip to domain
        X = np.clip(np.array(X), landscape.bounds[:, 0], landscape.bounds[:, 1])
        f = landscape(X)
        es.tell(X.tolist(), f.tolist())
        X_hist.extend(X.tolist()); f_hist.extend(f.tolist())
        n_evals += len(X)
    X_hist = np.array(X_hist[:budget]); f_hist = np.array(f_hist[:budget])
    basins = landscape.basin_of(X_hist)
    best_idx = int(np.argmin(f_hist))
    return {
        "method": "cmaes",
        "X": X_hist, "f": f_hist, "basins": basins,
        "best_final": float(f_hist[best_idx]),
        "best_point": X_hist[best_idx],
        "success": int(np.array_equal(X_hist[best_idx], landscape.mu_g) or
                       landscape.basin_of(X_hist[best_idx:best_idx+1])[0] == 0),
        "first_entry_index": int(np.argmax(basins == 0)) if (basins == 0).any() else None,
        "frac_global": float((basins == 0).mean()),
        "frac_local": float((basins == 1).mean()),
    }


def run_bo(landscape, budget=300, seed=0, n_init=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d = landscape.d
    n_init = n_init or (2 * d + 5)
    bounds = torch.tensor(landscape.bounds.T, dtype=torch.double)  # shape (2, d)

    sampler = qmc.LatinHypercube(d=d, seed=seed)
    X_init = torch.tensor(sampler.random(n_init) * 20 - 10, dtype=torch.double)
    Y_init = torch.tensor(landscape(X_init.numpy()), dtype=torch.double).unsqueeze(-1)

    X_all = X_init.clone(); Y_all = Y_init.clone()

    while X_all.shape[0] < budget:
        gp = SingleTaskGP(X_all, Y_all)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        ei = ExpectedImprovement(gp, best_f=Y_all.min().item(), maximize=False)
        cand, _ = optimize_acqf(
            ei, bounds=bounds, q=1, num_restarts=10, raw_samples=128
        )
        y_new = torch.tensor(landscape(cand.numpy()), dtype=torch.double).unsqueeze(-1)
        X_all = torch.cat([X_all, cand]); Y_all = torch.cat([Y_all, y_new])

    X_np = X_all.numpy(); f_np = Y_all.squeeze(-1).numpy()
    basins = landscape.basin_of(X_np)
    best_idx = int(np.argmin(f_np))
    return {
        "method": "bo",
        "X": X_np, "f": f_np, "basins": basins,
        "best_final": float(f_np[best_idx]),
        "best_point": X_np[best_idx],
        "success": int(basins[best_idx] == 0),
        "first_entry_index": int(np.argmax(basins == 0)) if (basins == 0).any() else None,
        "frac_global": float((basins == 0).mean()),
        "frac_local": float((basins == 1).mean()),
    }
