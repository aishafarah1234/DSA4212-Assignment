"""Fixed-generation population control: lambda in {6, 12, 24}, G = 70, one run per arm.

  python code/experiment4_fixedgen.py --workers 3
  python code/experiment4_fixedgen.py --aggregate-only

Isolates population width from adaptation time: every arm completes the same total
number of generations (G = 70) in a single run, so the arms differ only in lambda and
in the number of evaluations lambda * G. Not pooled with the 300-evaluation experiments.
"""
import argparse
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from math import sqrt

import numpy as np
import pandas as pd
from scipy.stats import binomtest, qmc
from objective import two_basin

LO, HI, DELTA, TOL = -5., 5., 1., .01
GENERATIONS = 70
POPULATIONS = (6, 12, 24)
METHODS = tuple(f'CMA-ES-G{GENERATIONS}-POP{P}' for P in POPULATIONS)


def obj(x, a, d):
    return float(two_basin(np.asarray(x), alpha=a, distance=d, delta=DELTA))


def region(x, a, d):
    x = np.asarray(x)
    decoy = np.sum((x - [-2., 0.]) ** 2)
    glob = a * np.sum((x - [d - 2., 0.]) ** 2) - DELTA
    return glob < decoy


class Recorder:
    def __init__(self, a, d):
        self.a, self.d = a, d
        self.best = float('inf')
        self.points, self.history = [], []
        self.first_deep = self.first_branch = None
        self.deep_evals = self.branch_evals = 0

    def add(self, x):
        x = np.asarray(x, dtype=float)
        y = obj(x, self.a, self.d)
        self.best = min(self.best, y)
        self.points.append(x.tolist())
        self.history.append(max(0., self.best + DELTA))
        n = len(self.points)
        if y < 0:
            self.deep_evals += 1
            if self.first_deep is None:
                self.first_deep = n
        if region(x, self.a, self.d):
            self.branch_evals += 1
            if self.first_branch is None:
                self.first_branch = n
        return y

    def finish(self):
        cells = {tuple(np.clip(np.floor(np.asarray(x) - LO), 0, 9).astype(int)) for x in self.points}
        return dict(best_value=self.best, final_regret=max(0., self.best + DELTA),
                    success=int(self.best + DELTA <= TOL), deep=int(self.best < 0),
                    branch=int(self.first_branch is not None),
                    first_deep_eval=self.first_deep, first_branch_eval=self.first_branch,
                    n_deep_evals=self.deep_evals, n_branch_evals=self.branch_evals,
                    grid_coverage=len(cells) / 100., evaluations_used=len(self.points),
                    history=self.history)


def cma_fixed_gen(a, d, seed, popsize, generations):
    import cma
    budget = popsize * generations
    rec = Recorder(a, d)
    x0 = np.random.default_rng(seed).uniform(LO, HI, size=2)
    es = cma.CMAEvolutionStrategy(x0, 2.0, {'bounds': [[LO, LO], [HI, HI]], 'seed': seed + 1,
                                            'popsize': popsize, 'verbose': -9, 'maxfevals': budget})
    gens_done = 0
    while not es.stop() and gens_done < generations and len(rec.points) < budget:
        xs = es.ask()
        ys = [rec.add(x) for x in xs]
        es.tell(xs, ys)
        gens_done += 1
    while len(rec.points) < budget:
        for x in es.ask(number=min(popsize, budget - len(rec.points))):
            rec.add(x)
    out = rec.finish()
    out.update(popsize=int(popsize), generations_target=int(generations),
               generations_completed=int(gens_done), budget=int(budget))
    return out


def run_one(job):
    a, d, method, seed, folder = job
    path = Path(folder) / 'runs' / f'a{a:g}_D{d:g}_{method}_s{seed}.json'
    if path.exists():
        return str(path), 'skipped'
    start = time.perf_counter()
    pop = int(method.rsplit('POP', 1)[1])
    r = cma_fixed_gen(a, d, seed, pop, GENERATIONS)
    r.update(alpha=a, distance=d, cell=f'a{a:g}_D{d:g}', acquisition=method,
             seed=seed, runtime_s=time.perf_counter() - start)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(r))
    tmp.replace(path)
    return str(path), 'done'


def wilson(k, n):
    z = 1.96
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    rad = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0., mid - rad), min(1., mid + rad)


def holm(pvalues):
    order = np.argsort(pvalues)
    out = [0.] * len(pvalues)
    prev = 0.
    for rank, ix in enumerate(order):
        prev = max(prev, (len(pvalues) - rank) * pvalues[ix])
        out[ix] = min(1., prev)
    return out


def aggregate(folder):
    folder = Path(folder)
    recs = [json.loads(p.read_text()) for p in sorted((folder / 'runs').glob('*.json'))]
    if not recs:
        raise RuntimeError('No saved runs found')
    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'history'} for r in recs])
    df.to_csv(folder / 'individual_runs.csv', index=False)

    summary = []
    for (cell, method), g in df.groupby(['cell', 'acquisition']):
        row = dict(cell=cell, alpha=g.alpha.iloc[0], distance=g.distance.iloc[0],
                   acquisition=method, n=len(g),
                   popsize=int(g.popsize.iloc[0]),
                   generations_target=int(g.generations_target.iloc[0]),
                   generations_completed_mean=float(g.generations_completed.mean()),
                   budget=int(g.budget.iloc[0]))
        for metric in ('deep', 'success', 'branch'):
            k = int(g[metric].sum())
            low, high = wilson(k, len(g))
            row.update({f'{metric}_k': k, f'{metric}_rate': k / len(g),
                        f'{metric}_lo': low, f'{metric}_hi': high})
        row.update(mean_final_regret=g.final_regret.mean(),
                   median_final_regret=g.final_regret.median(),
                   median_evals_used=g.evaluations_used.median(),
                   min_evals_used=int(g.evaluations_used.min()),
                   median_runtime_s=g.runtime_s.median())
        summary.append(row)
    summary = pd.DataFrame(summary).sort_values(['alpha', 'distance', 'popsize'])
    summary.to_csv(folder / 'summary.csv', index=False)
    print(summary.to_string(index=False))

    cma_arms = [m for m in METHODS if m.startswith('CMA-ES-')]
    comparisons = [(cma_arms[i], cma_arms[j], 'population')
                   for i in range(len(cma_arms))
                   for j in range(i + 1, len(cma_arms))]
    tests = []
    for metric in ('deep', 'success', 'branch'):
        group = []
        for cell, g in df[df.acquisition.isin(cma_arms)].groupby('cell'):
            for base_name, alt_name, family in comparisons:
                b = g[g.acquisition == base_name].set_index('seed')[metric].astype(bool)
                a = g[g.acquisition == alt_name].set_index('seed')[metric].astype(bool)
                seeds = b.index.intersection(a.index)
                if len(seeds) == 0:
                    continue
                bv, av = b.loc[seeds].to_numpy(), a.loc[seeds].to_numpy()
                alt_only = int(np.sum(~bv & av))
                base_only = int(np.sum(bv & ~av))
                discordant = alt_only + base_only
                p = binomtest(alt_only, discordant, .5).pvalue if discordant else 1.
                group.append(dict(metric=metric, family=family, cell=cell,
                                  baseline=base_name, alternative=alt_name,
                                  n_pairs=len(seeds), baseline_rate=bv.mean(), alt_rate=av.mean(),
                                  alt_only=alt_only, baseline_only=base_only, p=p))
        for x, adjusted in zip(group, holm([x['p'] for x in group])):
            x['holm_p'] = adjusted
        tests.extend(group)
    pd.DataFrame(tests).to_csv(folder / 'paired_tests.csv', index=False)
    print('\nPaired tests between CMA-ES arms (same seed, same total generations, different populations):')
    print(pd.DataFrame(tests).to_string(index=False))
    print('\nSaved outputs in', folder)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', nargs='+', default=['5:4', '10:4', '20:4', '10:1', '10:2', '10:3'])
    ap.add_argument('--runs', type=int, default=30)
    ap.add_argument('--seed-start', type=int, default=5000)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--output', type=Path, default=Path('results/experiment4_fixedgen'))
    ap.add_argument('--aggregate-only', action='store_true')
    args = ap.parse_args()
    if args.runs < 1 or args.workers < 1:
        ap.error('runs and workers must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'runs').mkdir(exist_ok=True)

    if not args.aggregate_only:
        cells = []
        for s in args.cells:
            a, d = map(float, s.split(':'))
            radius = sqrt(DELTA / a)
            if not (LO < d - 2 - radius and d - 2 + radius < HI):
                ap.error(f'Global negative-value region outside box: {s}')
            cells.append((a, d))
        jobs = [(a, d, m, args.seed_start + i, str(args.output))
                for a, d in cells for m in METHODS for i in range(args.runs)]
        print(f'{len(jobs)} runs (G={GENERATIONS}, populations={POPULATIONS}) -> {args.output}', flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_one, j) for j in jobs]
            for i, future in enumerate(as_completed(futures), 1):
                path, status = future.result()
                print(f'[{i}/{len(jobs)}] {Path(path).name}: {status}', flush=True)
    aggregate(args.output)


if __name__ == '__main__':
    main()