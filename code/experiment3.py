"""Experiment 3: effect of the BO acquisition function on two-basin landscapes.

One script serves both the width part and the separation part: they differ only in --cells.
Cells are alpha:distance pairs (delta = 1 and the box [-5, 5]^2 are fixed, as in Experiments 1-2).

  python code/experiment3.py --cells 10:1 10:4 --output results/experiment3_separation --workers 3
  python code/experiment3.py --cells 5:4 10:4 20:4 --output results/experiment3_width --workers 3
  python code/experiment3.py --output results/experiment3_width --aggregate-only   # rebuild tables/figures

Design (decide the cells and the primary metric BEFORE running, and write them in the report):
  * Baseline = Experiment 2's BO: SingleTaskGP + Standardize, 8 scrambled Sobol points, 5 restarts,
    64 raw samples, 200 evaluations. Only the acquisition function changes.
  * LogEI is RE-RUN here on the same fresh seeds as the alternatives, so every comparison is paired
    (same seed => same initial design) and the baseline was not used to select the seeds.
  * Fresh seeds (default 2000+) so Stage 2 never reuses Stage 1 seeds (1000+ in Exp 2, 0-29 in Exp 1).
  * Every run is saved to <output>/runs/*.json as soon as it finishes; reruns skip finished runs.
Outcomes per run: deep (found f < 0, i.e. the global basin's bottom region), success (regret <= 0.01),
branch (visited the region where the narrow bowl governs f), plus exploration diagnostics.
"""
import argparse
import itertools
import json
import os
import random
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from objective import two_basin

# ------------------------------------------------------------------ settings (same as Experiment 2)
DELTA = 1.0
DIM = 2
LOWER, UPPER = -5.0, 5.0
SUCCESS_TOL = 0.01
N_INITIAL = 8
NUM_RESTARTS = 5
RAW_SAMPLES = 64
DEFAULT_ACQS = ("LogEI", "LogPI", "UCB-0.5", "UCB-4")
BASELINE = "LogEI"
A_CENTRE = np.array([-2.0, 0.0])  # decoy minimum, as in two_basin()


# ------------------------------------------------------------------ objective helpers
def objective(x, alpha, distance):
    return two_basin(x, alpha=alpha, distance=distance, delta=DELTA)


def regret(value):
    return max(0.0, float(value) + DELTA)  # global minimum is -DELTA


def unnormalise(u):
    return LOWER + (UPPER - LOWER) * u


def in_narrow_region(x, alpha, distance):
    """True where the narrow bowl is the smaller component, i.e. f follows it."""
    b = A_CENTRE + np.array([distance, 0.0])
    x = np.asarray(x, float)
    return bool(alpha * np.sum((x - b) ** 2) - DELTA < np.sum((x - A_CENTRE) ** 2))


def check_cell(alpha, distance):
    r = np.sqrt(DELTA / alpha)  # radius of the f < 0 ball around the global centre
    bx = -2.0 + distance
    if not (LOWER <= bx - r and bx + r <= UPPER):
        raise ValueError(f"cell alpha={alpha}, distance={distance}: the f<0 ball leaves the box")


class Tracker:
    """Per-evaluation bookkeeping shared by every acquisition."""

    def __init__(self, alpha, distance):
        self.alpha, self.distance = alpha, distance
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
        if in_narrow_region(x, self.alpha, self.distance):
            self.n_branch += 1
            self.first_branch = self.first_branch or n

    def summary(self):
        cells = {tuple(np.clip(np.floor(p - LOWER), 0, 9).astype(int)) for p in self.points}
        if self.best < 0:
            basin = "global"
        elif np.linalg.norm(self.best_x - A_CENTRE) < 0.1:
            basin = "decoy"
        else:
            basin = "other"
        return dict(best_value=self.best, final_regret=regret(self.best),
                    success=int(regret(self.best) <= SUCCESS_TOL), deep=int(self.best < 0),
                    first_deep_eval=self.first_deep, first_branch_eval=self.first_branch,
                    n_deep_evals=self.n_deep, n_branch_evals=self.n_branch,
                    grid_coverage=len(cells) / 100.0,  # fraction of the 100 unit cells of the box visited
                    final_basin=basin, best_x0=float(self.best_x[0]), best_x1=float(self.best_x[1]),
                    evaluations_used=len(self.points))


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
        # BoTorch weights the posterior std by sqrt(beta); we maximise -f, i.e. a lower confidence bound on f
        return UpperConfidenceBound(model=model, beta=float(name.split("-")[1]), posterior_transform=negate)
    raise ValueError(f"unknown acquisition {name}")


def run_bo(alpha, distance, budget, seed, acq_name):
    import torch
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dtype = torch.double
    t0 = time.perf_counter()
    tracker = Tracker(alpha, distance)
    runtime_warnings = failed_opt = fit_warnings = fallbacks = 0

    sobol = torch.quasirandom.SobolEngine(DIM, scramble=True, seed=seed)  # depends on the seed only
    train_X = sobol.draw(min(N_INITIAL, budget)).to(dtype=dtype)
    ys = []
    for u in train_X:
        x = unnormalise(u).numpy()
        y = objective(x, alpha, distance)
        tracker.add(x, y)
        ys.append(y)
    train_Y = torch.tensor(ys, dtype=dtype).unsqueeze(-1)
    bounds = torch.stack([torch.zeros(DIM, dtype=dtype), torch.ones(DIM, dtype=dtype)])

    while len(tracker.points) < budget:
        try:
            model = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            with warnings.catch_warnings(record=True) as fit_caught:
                warnings.simplefilter("always")
                fit_gpytorch_mll(mll)
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
            fallbacks += 1  # a Sobol point replaces the BO choice; report this count
            print(f"fallback: alpha={alpha} D={distance} {acq_name} seed={seed} "
                  f"eval={len(tracker.points) + 1}: {exc}", flush=True)
            candidate = sobol.draw(1).to(dtype=dtype)
        x = unnormalise(candidate[0]).numpy()
        y = objective(x, alpha, distance)
        tracker.add(x, y)
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, torch.tensor([[y]], dtype=dtype)], dim=0)

    rec = tracker.summary()
    rec.update(runtime_s=time.perf_counter() - t0, runtime_warnings=runtime_warnings,
               failed_opt_msgs=failed_opt, fit_warnings=fit_warnings, fallback_count=fallbacks)
    return rec, tracker.history


# ------------------------------------------------------------------ jobs
def cell_label(alpha, distance):
    return f"a{alpha:g}_D{distance:g}"


def run_job(job):
    path = job["path"]
    if os.path.exists(path):
        return path, "skipped"
    try:
        import torch
        torch.set_num_threads(1)  # avoid oversubscription with several workers
    except ImportError:
        pass
    rec, history = run_bo(job["alpha"], job["distance"], job["budget"], job["seed"], job["acq"])
    rec.update(alpha=job["alpha"], distance=job["distance"], cell=cell_label(job["alpha"], job["distance"]),
               acquisition=job["acq"], seed=job["seed"], budget=job["budget"], history=history)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh)
    os.replace(tmp, path)  # atomic: a crash never leaves half a file
    return path, f"done in {rec['runtime_s']:.0f}s deep={rec['deep']} success={rec['success']}"


def valid_acquisition(name):
    """A typo would raise inside the BO step and silently turn the arm into Sobol fallbacks."""
    if name in ("LogEI", "LogPI"):
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
    """Exact McNemar test on paired binary outcomes (same seeds)."""
    from scipy.stats import binomtest
    b = int(np.sum(~base & alt))  # alternative succeeds, baseline fails
    c = int(np.sum(base & ~alt))  # baseline succeeds, alternative fails
    return (1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue), b, c


# ------------------------------------------------------------------ aggregation
def aggregate(out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = [json.load(open(p)) for p in sorted(Path(out_dir, "runs").glob("*.json"))]
    if not recs:
        raise SystemExit(f"no runs found in {out_dir}/runs")
    hist = pd.DataFrame([dict(cell=r["cell"], acquisition=r["acquisition"], seed=r["seed"], evaluation=t,
                              regret=v) for r in recs for t, v in enumerate(r["history"], 1)])
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "history"} for r in recs])
    if df.budget.nunique() > 1:
        raise SystemExit(f"runs with different budgets in {out_dir}: {sorted(int(b) for b in df.budget.unique())}; "
                         "use a separate output folder per budget")
    df["branch"] = df["first_branch_eval"].notna().astype(int)
    df.to_csv(Path(out_dir, "individual_runs.csv"), index=False)
    hist.to_csv(Path(out_dir, "convergence_raw.csv"), index=False)

    metrics = ["deep", "success", "branch"]
    rows = []
    for (cell, acq), g in df.groupby(["cell", "acquisition"]):
        row = dict(cell=cell, alpha=g.alpha.iloc[0], distance=g.distance.iloc[0], acquisition=acq, n=len(g))
        for m in metrics:
            k = int(g[m].sum())
            lo, hi = wilson(k, len(g))
            row.update({f"{m}_k": k, f"{m}_rate": k / len(g), f"{m}_lo": lo, f"{m}_hi": hi})
        row.update(median_final_regret=g.final_regret.median(), mean_final_regret=g.final_regret.mean(),
                   median_grid_coverage=g.grid_coverage.median(),
                   mean_branch_evals=g.n_branch_evals.mean(), mean_deep_evals=g.n_deep_evals.mean(),
                   median_runtime_s=g.runtime_s.median(), runtime_warnings=int(g.runtime_warnings.sum()),
                   failed_opt_msgs=int(g.failed_opt_msgs.sum()), fit_warnings=int(g.fit_warnings.sum()),
                   fallbacks=int(g.fallback_count.sum()))
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["alpha", "distance", "acquisition"])
    summary.to_csv(Path(out_dir, "summary.csv"), index=False)

    tests = []  # each alternative vs the LogEI baseline, paired by seed, per cell and metric
    for m in metrics:
        part = []
        for cell, g in df.groupby("cell"):
            base = g[g.acquisition == BASELINE].set_index("seed")[m].astype(bool)
            for acq in sorted(set(g.acquisition) - {BASELINE}):
                alt = g[g.acquisition == acq].set_index("seed")[m].astype(bool)
                seeds = base.index.intersection(alt.index)
                p, b, c = mcnemar_exact(base[seeds].to_numpy(), alt[seeds].to_numpy())
                part.append(dict(metric=m, cell=cell, alternative=acq, n_pairs=len(seeds),
                                 baseline_rate=base[seeds].mean(), alt_rate=alt[seeds].mean(),
                                 alt_only=b, baseline_only=c, p=p))
        if part:
            adj = holm([t["p"] for t in part])
            for t, a in zip(part, adj):
                t["holm_p"] = a
            tests += part
    pd.DataFrame(tests).to_csv(Path(out_dir, "paired_tests_vs_logei.csv"), index=False)
    pd.crosstab([df.cell, df.acquisition], df.final_basin).to_csv(Path(out_dir, "final_basin_counts.csv"))
    print(summary.to_string(index=False))
    print("\nPaired tests vs", BASELINE, "(exact McNemar, Holm within each metric):")
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
                   yerr=[s[f"{m}_rate"] - s[f"{m}_lo"], s[f"{m}_hi"] - s[f"{m}_rate"]], capsize=2)
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
        ax.set_title(cell, fontsize=9)
        ax.set_xlabel("objective evaluations")
    axes[0][0].set_ylabel("fraction of runs with f < 0 found")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(Path(out_dir, "deep_discovery_vs_evaluations.png"), dpi=200)
    plt.close(fig)
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
    args = ap.parse_args(argv)

    if not args.aggregate_only:
        if not args.cells:
            ap.error("--cells is required unless --aggregate-only")
        if args.runs < 1 or args.workers < 1:
            ap.error("--runs and --workers must be at least 1")
        if args.budget <= N_INITIAL:
            ap.error(f"--budget must exceed the {N_INITIAL} initial points")
        bad = [a for a in args.acquisitions if not valid_acquisition(a)]
        if bad:
            ap.error(f"unknown acquisition(s) {bad}; use LogEI, LogPI or UCB-<beta>")
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
        cells = list(dict.fromkeys(cells))  # drop duplicate cells
        for a, d in cells:
            try:
                check_cell(a, d)
            except ValueError as exc:
                ap.error(str(exc))

        # settings that must not change inside one output folder, or old and new runs would be mixed
        fixed = dict(budget=args.budget, delta=DELTA, success_tol=SUCCESS_TOL, n_initial=N_INITIAL,
                     num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES, lower=LOWER, upper=UPPER)
        runs_dir = args.output / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = args.output / "config_used.json"
        old = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        if old.get("fixed") and old["fixed"] != fixed:
            ap.error(f"{args.output} was created with different settings {old['fixed']}; "
                     "use a new --output folder (old and new runs must not be mixed)")
        invocation = dict(cells=args.cells, acquisitions=args.acquisitions, runs=args.runs,
                          seed_start=args.seed_start, workers=args.workers)
        cfg_path.write_text(json.dumps(
            dict(fixed=fixed, invocations=old.get("invocations", []) + [invocation]), indent=2))

        jobs = [dict(alpha=a, distance=d, acq=acq, seed=args.seed_start + s, budget=args.budget,
                     path=str(runs_dir / f"{cell_label(a, d)}_{acq}_s{args.seed_start + s}.json"))
                for (a, d), acq, s in itertools.product(cells, args.acquisitions, range(args.runs))]
        print(f"{len(jobs)} BO runs -> {args.output}  (finished runs are skipped)", flush=True)
        with Pool(args.workers) as pool:
            for i, (path, status) in enumerate(pool.imap_unordered(run_job, jobs), 1):
                print(f"[{i}/{len(jobs)}] {os.path.basename(path)}: {status}", flush=True)
    aggregate(args.output)


if __name__ == "__main__":
    main()
