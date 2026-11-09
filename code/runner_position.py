"""Shared runner for Experiments 1-3 with three basin layouts.

Same as runner.py except the objective geometry is selected by `layout`:
  original        a = (-2, 0),       b = (-2 + d, 0)
  fixed_global    a = (-d, 0),       b = (0, 0)
  fixed_midpoint  a = (-d/2, 0),     b = (+d/2, 0)

`layout` is passed explicitly through every function; there is no module-global.
`check_cell` validates both centres and a boundary margin (MARGIN); the margin is recorded
in config_used.json. `aggregate` refuses to mix layouts in one output folder and reports
under-spent CMA-ES runs via n_underspent / median_evaluation_padding.

  python code/experiment2_position.py --layout fixed_global --workers 3
"""
import argparse
import itertools
import json
import os
import random
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ settings
DELTA = 1.0
DIM = 2
LOWER, UPPER = -5.0, 5.0
MARGIN = 1.0               # experimental design constraint: both basin centres must be at least this far
                           # from the box boundary. Chosen so the tested cells (alpha=10, d<=4) are admissible
                           # in all three layouts. Not a property of two_basin.
SUCCESS_TOL = 0.01
N_INITIAL = 8
NUM_RESTARTS = 5
RAW_SAMPLES = 64
DEFAULT_ACQS = ("LogEI", "LogPI", "UCB-0.5", "UCB-4")
BASELINE = "LogEI"
PILOT_N_INITIAL = 10
REFERENCE_ARMS = ("Random", "Sobol")
LAYOUTS = ("original", "fixed_global", "fixed_midpoint")


def basin_centres(distance, layout):
    """Local (a) and global (b) centres for the given layout."""
    d = float(distance)
    if layout == "original":
        return np.array([-2.0, 0.0]), np.array([-2.0 + d, 0.0])
    if layout == "fixed_global":
        return np.array([-d, 0.0]), np.array([0.0, 0.0])
    if layout == "fixed_midpoint":
        return np.array([-d / 2.0, 0.0]), np.array([d / 2.0, 0.0])
    raise ValueError(f"unknown layout: {layout}")


def cma_start(distance, layout):
    """CMA-ES start relative to the local basin: start at a in every layout.
    run_cma_pilot ignores this and draws a uniform random mean, for continuity with the notebook."""
    a, _ = basin_centres(distance, layout)
    return a.astype(float)


# ------------------------------------------------------------------ objective helpers
def objective(x, alpha, distance, layout):
    a, b = basin_centres(distance, layout)
    x = np.asarray(x, dtype=float)
    return float(min(np.sum((x - a) ** 2), alpha * np.sum((x - b) ** 2) - DELTA))


def regret(value):
    return max(0.0, float(value) + DELTA)


def unnormalise(u):
    return LOWER + (UPPER - LOWER) * u


def in_narrow_region(x, alpha, distance, layout):
    a, b = basin_centres(distance, layout)
    x = np.asarray(x, float)
    return bool(alpha * np.sum((x - b) ** 2) - DELTA < np.sum((x - a) ** 2))


def check_cell(alpha, distance, layout):
    """Both centres inside the box and at least MARGIN from the boundary; global f<0 ball fully inside."""
    a, b = basin_centres(distance, layout)
    for name, c in (("local", a), ("global", b)):
        if not (LOWER <= c[0] <= UPPER and LOWER <= c[1] <= UPPER):
            raise ValueError(f"layout={layout} cell alpha={alpha}, d={distance}: {name} centre {c} outside the box")
        m = min(c[0] - LOWER, UPPER - c[0], c[1] - LOWER, UPPER - c[1])
        if m < MARGIN:
            raise ValueError(f"layout={layout} cell alpha={alpha}, d={distance}: {name} centre {c} has margin "
                             f"{m:.2f} < {MARGIN}")
    r = np.sqrt(DELTA / alpha)
    if not (LOWER <= b[0] - r and b[0] + r <= UPPER):
        raise ValueError(f"layout={layout} cell alpha={alpha}, d={distance}: global f<0 ball leaves the box")


class Tracker:
    def __init__(self, alpha, distance, layout):
        self.alpha, self.distance, self.layout = alpha, distance, layout
        self.points, self.history = [], []
        self.best, self.best_x = float("inf"), None
        self.first_deep = self.first_branch = None
        self.n_deep = self.n_branch = 0

    def add(self, x, y):
        x = np.asarray(x, float)
        self.points.append(x)
        n = len(self.points)
        if y < self.best:
            self.best, self.best_x = float(y), x
        self.history.append(regret(self.best))
        if y < 0:
            self.n_deep += 1
            self.first_deep = self.first_deep or n
        if in_narrow_region(x, self.alpha, self.distance, self.layout):
            self.n_branch += 1
            self.first_branch = self.first_branch or n

    def summary(self):
        cells = {tuple(np.clip(np.floor(p - LOWER), 0, 9).astype(int)) for p in self.points}
        a, _ = basin_centres(self.distance, self.layout)
        if self.best < 0:
            basin = "global"
        elif np.linalg.norm(self.best_x - a) < 0.1:
            basin = "decoy"
        else:
            basin = "other"
        return dict(best_value=self.best, final_regret=regret(self.best),
                    success=int(regret(self.best) <= SUCCESS_TOL), deep=int(self.best < 0),
                    first_deep_eval=self.first_deep, first_branch_eval=self.first_branch,
                    n_deep_evals=self.n_deep, n_branch_evals=self.n_branch,
                    grid_coverage=len(cells) / 100.0,
                    final_basin=basin, best_x0=float(self.best_x[0]), best_x1=float(self.best_x[1]),
                    evaluations_used=len(self.points),
                    points=[p.tolist() for p in self.points])


# ------------------------------------------------------------------ Bayesian optimisation
def build_acquisition(name, model, best_f):
    import torch
    from botorch.acquisition.analytic import (LogExpectedImprovement, LogProbabilityOfImprovement,
                                              UpperConfidenceBound)
    if name == "LogEI":
        return LogExpectedImprovement(model=model, best_f=best_f, maximize=False)
    if name == "LogPI":
        return LogProbabilityOfImprovement(model=model, best_f=best_f, maximize=False)
    if name.startswith("UCB-"):
        from botorch.acquisition.objective import ScalarizedPosteriorTransform
        negate = ScalarizedPosteriorTransform(weights=torch.tensor([-1.0], dtype=torch.double))
        return UpperConfidenceBound(model=model, beta=float(name.split("-")[1]), posterior_transform=negate)
    raise ValueError(f"unknown acquisition {name}")


def run_bo(alpha, distance, budget, seed, acq_name, layout):
    import torch
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dtype = torch.double
    t0 = time.perf_counter()
    tracker = Tracker(alpha, distance, layout)
    runtime_warnings = failed_opt = fit_warnings = fallbacks = 0

    sobol = torch.quasirandom.SobolEngine(DIM, scramble=True, seed=seed)
    train_X = sobol.draw(min(N_INITIAL, budget)).to(dtype=dtype)
    ys = []
    for u in train_X:
        x = unnormalise(u).numpy()
        y = objective(x, alpha, distance, layout)
        tracker.add(x, y); ys.append(y)
    train_Y = torch.tensor(ys, dtype=dtype).unsqueeze(-1)
    bounds = torch.stack([torch.zeros(DIM, dtype=dtype), torch.ones(DIM, dtype=dtype)])

    while len(tracker.points) < budget:
        try:
            model = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            with warnings.catch_warnings(record=True) as fit_caught:
                warnings.simplefilter("always"); fit_gpytorch_mll(mll)
            fit_warnings += len(fit_caught)
            acq = build_acquisition(acq_name, model, train_Y.min().item())
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", RuntimeWarning)
                candidate, _ = optimize_acqf(acq_function=acq, bounds=bounds, q=1,
                                             num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES)
            runtime_warnings += sum(issubclass(w.category, RuntimeWarning) for w in caught)
            failed_opt += sum("Optimization failed" in str(w.message) for w in caught)
            candidate = candidate.detach()
            if not torch.isfinite(candidate).all():
                raise ValueError("non-finite acquisition candidate")
        except Exception as exc:
            fallbacks += 1
            print(f"fallback: layout={layout} alpha={alpha} D={distance} {acq_name} seed={seed} "
                  f"eval={len(tracker.points) + 1}: {exc}", flush=True)
            candidate = sobol.draw(1).to(dtype=dtype)
        x = unnormalise(candidate[0]).numpy()
        y = objective(x, alpha, distance, layout)
        tracker.add(x, y)
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, torch.tensor([[y]], dtype=dtype)], dim=0)

    rec = tracker.summary()
    rec.update(runtime_s=time.perf_counter() - t0, runtime_warnings=runtime_warnings,
               failed_opt_msgs=failed_opt, fit_warnings=fit_warnings, fallback_count=fallbacks,
               evaluation_padding=0)
    return rec, tracker.history


# ------------------------------------------------------------------ non-BO arms
NON_BO = ("CMA-ES", "Random", "Sobol")


def run_cma(alpha, distance, budget, seed, layout):
    """CMA-ES started at the local basin centre a, sigma0 = 1. Residual evaluations are Sobol-padded
    so evaluations_used == budget; the padding is recorded in evaluation_padding."""
    import cma
    t0 = time.perf_counter()
    tracker = Tracker(alpha, distance, layout)
    x0 = cma_start(distance, layout)
    es = cma.CMAEvolutionStrategy(x0, 1.0, {"seed": seed, "bounds": [LOWER, UPPER], "verbose": -9})
    padding = 0
    while len(tracker.points) < budget and not es.stop():
        sols = es.ask()
        n = min(len(sols), budget - len(tracker.points))
        vals = []
        for x in sols[:n]:
            y = objective(x, alpha, distance, layout)
            vals.append(y); tracker.add(x, y)
        if n < len(sols):
            break
        es.tell(sols, vals)
    if len(tracker.points) < budget:
        pad_n = budget - len(tracker.points)
        pad_rng = np.random.default_rng(np.random.SeedSequence([seed, 9911]))
        for u in pad_rng.random((pad_n, DIM)):
            x = unnormalise(u)
            tracker.add(x, objective(x, alpha, distance, layout))
        padding = pad_n
    rec = tracker.summary()
    rec.update(runtime_s=time.perf_counter() - t0, runtime_warnings=0, failed_opt_msgs=0, fit_warnings=0,
               fallback_count=0, evaluation_padding=padding)
    return rec, tracker.history


def run_cma_pilot(alpha, distance, budget, seed, layout):
    """CMA-ES as in the original notebook: uniform-random start, sigma0 = 2, pycma seed = seed + 1,
    residual evaluations sampled from the final distribution without tell(). Layout-agnostic start."""
    import cma
    t0 = time.perf_counter()
    tracker = Tracker(alpha, distance, layout)
    x0 = np.random.default_rng(seed).uniform(LOWER, UPPER, size=2)
    es = cma.CMAEvolutionStrategy(x0, 2.0, {"bounds": [[LOWER, LOWER], [UPPER, UPPER]], "seed": seed + 1,
                                            "verbose": -9, "maxfevals": budget})
    while not es.stop():
        if len(tracker.points) + es.popsize > budget:
            break
        sols = es.ask()
        vals = [objective(x, alpha, distance, layout) for x in sols]
        es.tell(sols, vals)
        for x, y in zip(sols, vals):
            tracker.add(x, y)
    remaining = budget - len(tracker.points)
    if remaining > 0:
        for x in es.ask(number=remaining):
            tracker.add(x, objective(x, alpha, distance, layout))
    rec = tracker.summary()
    rec.update(runtime_s=time.perf_counter() - t0, runtime_warnings=0, failed_opt_msgs=0, fit_warnings=0,
               fallback_count=0, evaluation_padding=0)
    return rec, tracker.history


def run_bo_notebook(alpha, distance, budget, seed, acq_name, layout, n_initial=PILOT_N_INITIAL):
    import torch
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.input import Normalize
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dtype = torch.double
    t0 = time.perf_counter()
    tracker = Tracker(alpha, distance, layout)
    runtime_warnings = failed_opt = fit_warnings = fallbacks = 0

    bounds = torch.tensor([[LOWER] * DIM, [UPPER] * DIM], dtype=dtype)
    sobol = torch.quasirandom.SobolEngine(DIM, scramble=True, seed=seed)
    X = bounds[0] + (bounds[1] - bounds[0]) * sobol.draw(min(n_initial, budget)).to(dtype=dtype)
    ys = []
    for x_t in X:
        x = x_t.numpy()
        y = objective(x, alpha, distance, layout)
        tracker.add(x, y); ys.append(y)
    Y = torch.tensor(ys, dtype=dtype).unsqueeze(-1)

    while len(tracker.points) < budget:
        try:
            y_scaled = (Y - Y.mean()) / Y.std().clamp_min(1e-8)
            model = SingleTaskGP(train_X=X, train_Y=y_scaled, input_transform=Normalize(d=DIM, bounds=bounds))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            with warnings.catch_warnings(record=True) as fit_caught:
                warnings.simplefilter("always"); fit_gpytorch_mll(mll)
            fit_warnings += len(fit_caught)
            acq = build_acquisition(acq_name, model, y_scaled.min().item())
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", RuntimeWarning)
                candidate, _ = optimize_acqf(acq_function=acq, bounds=bounds, q=1,
                                             num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES)
            runtime_warnings += sum(issubclass(w.category, RuntimeWarning) for w in caught)
            failed_opt += sum("Optimization failed" in str(w.message) for w in caught)
            candidate = candidate.detach()
            if not torch.isfinite(candidate).all():
                raise ValueError("non-finite acquisition candidate")
        except Exception as exc:
            fallbacks += 1
            print(f"fallback: layout={layout} alpha={alpha} D={distance} {acq_name} seed={seed} "
                  f"eval={len(tracker.points) + 1}: {exc}", flush=True)
            candidate = bounds[0] + (bounds[1] - bounds[0]) * sobol.draw(1).to(dtype=dtype)
        x = candidate[0].numpy()
        y = objective(x, alpha, distance, layout)
        tracker.add(x, y)
        X = torch.cat([X, candidate], dim=0)
        Y = torch.cat([Y, torch.tensor([[y]], dtype=dtype)], dim=0)

    rec = tracker.summary()
    rec.update(runtime_s=time.perf_counter() - t0, runtime_warnings=runtime_warnings,
               failed_opt_msgs=failed_opt, fit_warnings=fit_warnings, fallback_count=fallbacks,
               evaluation_padding=0)
    return rec, tracker.history


def run_baseline(alpha, distance, budget, seed, kind, layout):
    t0 = time.perf_counter()
    tracker = Tracker(alpha, distance, layout)
    if kind == "Random":
        U = np.random.default_rng(seed).random((budget, DIM))
    else:
        from scipy.stats import qmc
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            U = qmc.Sobol(DIM, scramble=True, seed=seed).random(budget)
    for u in U:
        x = unnormalise(u)
        tracker.add(x, objective(x, alpha, distance, layout))
    rec = tracker.summary()
    rec.update(runtime_s=time.perf_counter() - t0, runtime_warnings=0, failed_opt_msgs=0, fit_warnings=0,
               fallback_count=0, evaluation_padding=0)
    return rec, tracker.history


def run_method(job):
    name, setup = job["acq"], job.get("setup", "aligned")
    a, d, b, sd, layout = job["alpha"], job["distance"], job["budget"], job["seed"], job["layout"]
    if name == "CMA-ES":
        return (run_cma_pilot(a, d, b, sd, layout) if setup == "pilot"
                else run_cma(a, d, b, sd, layout))
    if name in ("Random", "Sobol"):
        return run_baseline(a, d, b, sd, name, layout)
    if setup == "pilot":
        return run_bo_notebook(a, d, b, sd, name, layout)
    return run_bo(a, d, b, sd, name, layout)


# ------------------------------------------------------------------ jobs
def cell_label(alpha, distance):
    return f"a{alpha:g}_D{distance:g}"


def run_job(job):
    path = job["path"]
    if os.path.exists(path):
        return path, "skipped"
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    rec, history = run_method(job)
    rec.update(layout=job["layout"], alpha=job["alpha"], distance=job["distance"],
               cell=cell_label(job["alpha"], job["distance"]),
               acquisition=job["acq"], seed=job["seed"], budget=job["budget"], history=history)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh)
    os.replace(tmp, path)
    return path, f"done in {rec['runtime_s']:.0f}s deep={rec['deep']} success={rec['success']}"


def valid_acquisition(name):
    if name in ("LogEI", "LogPI") + NON_BO:
        return True
    if name.startswith("UCB-"):
        try:
            return float(name[4:]) > 0
        except ValueError:
            return False
    return False


# ------------------------------------------------------------------ statistics
def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return max(0.0, c - h), min(1.0, c + h)


def holm(p):
    p = np.asarray(p, float)
    order, adj, run = np.argsort(p), np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def mcnemar_exact(base, alt):
    from scipy.stats import binomtest
    b = int(np.sum(~base & alt))
    c = int(np.sum(base & ~alt))
    return (1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue), b, c


def plot_points(out_dir, recs, layout, n_seeds=3):
    import matplotlib.pyplot as plt
    for cell in dict.fromkeys(r["cell"] for r in recs):
        sub = [r for r in recs if r["cell"] == cell and r.get("points")]
        if not sub:
            continue
        acqs = list(dict.fromkeys(r["acquisition"] for r in sub))
        seeds = sorted({r["seed"] for r in sub})[:n_seeds]
        alpha, dist = sub[0]["alpha"], sub[0]["distance"]
        g = np.linspace(LOWER, UPPER, 200)
        X, Y = np.meshgrid(g, g)
        a, b = basin_centres(dist, layout)
        Z = np.minimum((X - a[0]) ** 2 + (Y - a[1]) ** 2,
                       alpha * ((X - b[0]) ** 2 + (Y - b[1]) ** 2) - DELTA)
        fig, axes = plt.subplots(len(acqs), len(seeds), figsize=(3.2 * len(seeds), 3.0 * len(acqs)), squeeze=False)
        for i, acq in enumerate(acqs):
            for j, sd in enumerate(seeds):
                ax = axes[i][j]
                ax.contour(X, Y, Z, levels=[-0.5, 0, 1, 5, 20], colors="gray", linewidths=0.4)
                r = next((r for r in sub if r["acquisition"] == acq and r["seed"] == sd), None)
                if r is not None:
                    p = np.array(r["points"])
                    ax.scatter(p[:, 0], p[:, 1], c=np.arange(len(p)), s=7, cmap="viridis")
                ax.plot(a[0], a[1], "k+", ms=8)
                ax.plot(b[0], b[1], "r+", ms=8)
                ax.set_xlim(LOWER, UPPER); ax.set_ylim(LOWER, UPPER)
                ax.set_title(f"{acq}, seed {sd}", fontsize=8)
        fig.tight_layout()
        fig.savefig(Path(out_dir, f"points_{cell}.png"), dpi=150)
        plt.close(fig)


# ------------------------------------------------------------------ aggregation
def aggregate(out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = [json.load(open(p)) for p in sorted(Path(out_dir, "runs").glob("*.json"))]
    if not recs:
        raise SystemExit(f"no runs found in {out_dir}/runs")
    layouts = {r.get("layout") for r in recs}
    if len(layouts) != 1 or None in layouts:
        raise SystemExit(f"mixed or missing layouts in {out_dir}: {sorted(l for l in layouts if l)}; "
                         "use a separate output folder per layout")
    layout = next(iter(layouts))

    hist = pd.DataFrame([dict(cell=r["cell"], acquisition=r["acquisition"], seed=r["seed"], evaluation=t,
                              regret=v) for r in recs for t, v in enumerate(r["history"], 1)])
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("history", "points")} for r in recs])
    if df.budget.nunique() > 1:
        raise SystemExit(f"runs with different budgets in {out_dir}: {sorted(int(b) for b in df.budget.unique())}; "
                         "use a separate output folder per budget")
    df["branch"] = df["first_branch_eval"].notna().astype(int)
    df.to_csv(Path(out_dir, "individual_runs.csv"), index=False)
    hist.to_csv(Path(out_dir, "convergence_raw.csv"), index=False)

    metrics = ["deep", "success", "branch"]
    rows = []
    for (cell, acq), g in df.groupby(["cell", "acquisition"]):
        row = dict(cell=cell, alpha=g.alpha.iloc[0], distance=g.distance.iloc[0], layout=layout,
                   acquisition=acq, n=len(g))
        for m in metrics:
            k = int(g[m].sum())
            lo, hi = wilson(k, len(g))
            row.update({f"{m}_k": k, f"{m}_rate": k / len(g), f"{m}_lo": lo, f"{m}_hi": hi})
        row.update(median_evals_used=g.evaluations_used.median(), min_evals_used=int(g.evaluations_used.min()),
                   n_underspent=int((g.evaluations_used < int(g.budget.iloc[0])).sum()),
                   median_evaluation_padding=g.evaluation_padding.median(),
                   median_final_regret=g.final_regret.median(), mean_final_regret=g.final_regret.mean(),
                   median_grid_coverage=g.grid_coverage.median(),
                   mean_branch_evals=g.n_branch_evals.mean(), mean_deep_evals=g.n_deep_evals.mean(),
                   median_runtime_s=g.runtime_s.median(), runtime_warnings=int(g.runtime_warnings.sum()),
                   failed_opt_msgs=int(g.failed_opt_msgs.sum()), fit_warnings=int(g.fit_warnings.sum()),
                   fallbacks=int(g.fallback_count.sum()))
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["alpha", "distance", "acquisition"])
    summary.to_csv(Path(out_dir, "summary.csv"), index=False)

    tests = []
    for m in metrics:
        part = []
        for cell, g in df.groupby("cell"):
            base = g[g.acquisition == BASELINE].set_index("seed")[m].astype(bool)
            for acq in sorted(set(g.acquisition) - {BASELINE}):
                alt = g[g.acquisition == acq].set_index("seed")[m].astype(bool)
                seeds = base.index.intersection(alt.index)
                p, b, c = mcnemar_exact(base[seeds].to_numpy(), alt[seeds].to_numpy())
                part.append(dict(metric=m, family="reference" if acq in REFERENCE_ARMS else "primary",
                                 cell=cell, alternative=acq, n_pairs=len(seeds),
                                 baseline_rate=base[seeds].mean(), alt_rate=alt[seeds].mean(),
                                 alt_only=b, baseline_only=c, p=p))
        for fam in ("primary", "reference"):
            sub = [t for t in part if t["family"] == fam]
            if sub:
                for t, a in zip(sub, holm([t["p"] for t in sub])):
                    t["holm_p"] = a
                tests += sub
    pd.DataFrame(tests).to_csv(Path(out_dir, "paired_tests_vs_logei.csv"), index=False)
    pd.crosstab([df.cell, df.acquisition], df.final_basin).to_csv(Path(out_dir, "final_basin_counts.csv"))
    print(summary.to_string(index=False))
    print(f"\nlayout = {layout}")
    print("Paired tests vs", BASELINE, "(exact McNemar; Holm within each metric and family; "
          "reference family = Random/Sobol):")
    print(pd.DataFrame(tests).to_string(index=False))

    cells = list(summary.cell.drop_duplicates())
    acqs = list(summary.acquisition.drop_duplicates())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    w = 0.8 / len(acqs)
    for ax, m in zip(axes, metrics):
        for j, acq in enumerate(acqs):
            s = summary[summary.acquisition == acq].set_index("cell").reindex(cells)
            x = np.arange(len(cells)) + (j - (len(acqs) - 1) / 2) * w
            ax.bar(x, s[f"{m}_rate"], w, label=acq,
                   yerr=np.clip([s[f"{m}_rate"] - s[f"{m}_lo"], s[f"{m}_hi"] - s[f"{m}_rate"]], 0, None),
                   capsize=2)
        ax.set_xticks(np.arange(len(cells)), cells, fontsize=8)
        ax.set_title({"deep": "deep discovery (f < 0)", "success": f"success (regret <= {SUCCESS_TOL})",
                      "branch": "visited narrow-bowl region"}[m], fontsize=9)
    axes[0].set_ylabel("rate over seeds (95% Wilson CI)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(Path(out_dir, "rates_by_acquisition.png"), dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(cells), figsize=(4.2 * len(cells), 3.6), sharey=True, squeeze=False)
    for ax, cell in zip(axes[0], cells):
        for acq in acqs:
            g = df[(df.cell == cell) & (df.acquisition == acq)]
            t = np.arange(1, int(g.budget.iloc[0]) + 1)
            ax.step(t, [(g.first_deep_eval.fillna(np.inf) <= i).mean() for i in t], where="post", label=acq)
        ax.set_title(cell, fontsize=9); ax.set_xlabel("objective evaluations")
    axes[0][0].set_ylabel("fraction of runs with f < 0 found")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(Path(out_dir, "deep_discovery_vs_evaluations.png"), dpi=200)
    plt.close(fig)
    plot_points(out_dir, recs, layout)
    print("\nSaved tables and figures to", Path(out_dir).resolve())


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", help="alpha:distance pairs, e.g. 10:1 10:4")
    ap.add_argument("--acquisitions", nargs="+", default=list(DEFAULT_ACQS))
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--seed-start", type=int, default=2000)
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--setup", choices=["aligned", "pilot"], default="aligned")
    ap.add_argument("--layout", choices=list(LAYOUTS), default="fixed_midpoint")
    args = ap.parse_args(argv)

    if not args.aggregate_only:
        if not args.cells:
            ap.error("--cells is required unless --aggregate-only")
        if args.runs < 1 or args.workers < 1:
            ap.error("--runs and --workers must be at least 1")
        n_init = PILOT_N_INITIAL if args.setup == "pilot" else N_INITIAL
        if args.budget <= n_init:
            ap.error(f"--budget must exceed the {n_init} initial points")
        bad = [a for a in args.acquisitions if not valid_acquisition(a)]
        if bad:
            ap.error(f"unknown acquisition(s) {bad}")
        if len(set(args.acquisitions)) != len(args.acquisitions):
            ap.error("duplicate acquisitions")
        if BASELINE not in args.acquisitions:
            ap.error(f"include {BASELINE}: every comparison is against it")
        try:
            cells = [tuple(float(v) for v in c.split(":")) for c in args.cells]
            if any(len(c) != 2 or c[0] <= 1e-9 or c[1] <= 0 for c in cells):
                raise ValueError
        except ValueError:
            ap.error("--cells must be alpha:distance pairs with alpha > 0 and distance > 0, e.g. 10:4")
        cells = list(dict.fromkeys(cells))
        for a, d in cells:
            try:
                check_cell(a, d, args.layout)
            except ValueError as exc:
                ap.error(str(exc))

        fixed = dict(layout=args.layout, margin=MARGIN, budget=args.budget, delta=DELTA,
                     success_tol=SUCCESS_TOL, n_initial=N_INITIAL, num_restarts=NUM_RESTARTS,
                     raw_samples=RAW_SAMPLES, lower=LOWER, upper=UPPER)
        if args.setup == "pilot":
            fixed.update(setup="pilot", n_initial=PILOT_N_INITIAL)
        runs_dir = args.output / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = args.output / "config_used.json"
        old = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        if old.get("fixed") and old["fixed"] != fixed:
            ap.error(f"{args.output} was created with different settings {old['fixed']}; "
                     "use a new --output folder (old and new runs must not be mixed)")
        if not cfg_path.exists() and any(runs_dir.glob("*.json")):
            ap.error(f"{args.output} already contains runs but has no config_used.json; it was produced by an "
                     "older runner. Use a new --output folder (e.g. add a _v2 suffix).")
        invocation = dict(layout=args.layout, cells=args.cells, acquisitions=args.acquisitions,
                          runs=args.runs, seed_start=args.seed_start, workers=args.workers, setup=args.setup)
        cfg_path.write_text(json.dumps(
            dict(fixed=fixed, invocations=old.get("invocations", []) + [invocation]), indent=2))

        jobs = [dict(alpha=a, distance=d, acq=acq, seed=args.seed_start + s, budget=args.budget,
                     setup=args.setup, layout=args.layout,
                     path=str(runs_dir / f"{cell_label(a, d)}_{acq}_s{args.seed_start + s}.json"))
                for (a, d), acq, s in itertools.product(cells, args.acquisitions, range(args.runs))]
        print(f"{len(jobs)} runs -> {args.output}  (finished runs are skipped)", flush=True)
        with Pool(args.workers) as pool:
            for i, (path, status) in enumerate(pool.imap_unordered(run_job, jobs), 1):
                print(f"[{i}/{len(jobs)}] {os.path.basename(path)}: {status}", flush=True)
    aggregate(args.output)


def run_with_defaults(defaults, argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    for flag, values in defaults.items():
        if flag not in argv:
            argv += [flag] + [str(v) for v in values]
    main(argv)


if __name__ == "__main__":
    main()