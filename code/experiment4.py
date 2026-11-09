"""Standalone Experiment 4: scheduled CMA-ES restarts, population separated from generations per restart.

Clean run (generation-matched, unclipped):
  python code/experiment4.py --budget 420 --restart-interval 60 --workers 3
Continuity run at budget 300 (IPOP-G / RR-S are clipped; flagged, not silently used):
  python code/experiment4.py --budget 300 --restart-interval 60 --allow-clipped --workers 3

Arms (all share the first CMA-ES segment: same start, step 2, default population, pycma seed):
  CMA-ES         no restarts
  CMA-ES-RR      restart every `interval`, new random start, population fixed
  CMA-ES-IPOP    as RR, population doubled at each restart
  CMA-ES-IPOP-G  population doubled, run length = pop * gens so every run gets the same generations
  CMA-ES-RR-S    population fixed, run lengths identical to IPOP-G (same schedule, more generations)
  Random, Sobol  chance-level references

The doubling schedules fit exactly when budget = interval * (2**k - 1) (420 for interval 60).
At any other budget, IPOP-G and RR-S are clipped; the script refuses to run them unless
--allow-clipped is passed, and records schedule_clipped per run.
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
BASE_POP = 6
METHODS = ('CMA-ES', 'CMA-ES-RR', 'CMA-ES-IPOP', 'CMA-ES-IPOP-G', 'CMA-ES-RR-S', 'Random', 'Sobol')
G_ARMS = ('CMA-ES-IPOP-G', 'CMA-ES-RR-S')


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


def run_schedule(method, budget, interval):
    gens = interval // BASE_POP
    runs, used, r = [], 0, 0
    while used < budget:
        if method == 'CMA-ES':
            pop, length = BASE_POP, budget
        elif method == 'CMA-ES-RR':
            pop, length = BASE_POP, interval
        elif method == 'CMA-ES-IPOP':
            pop, length = BASE_POP * 2 ** r, interval
        elif method == 'CMA-ES-IPOP-G':
            pop = BASE_POP * 2 ** r
            length = pop * gens
        elif method == 'CMA-ES-RR-S':
            pop, length = BASE_POP, BASE_POP * 2 ** r * gens
        else:
            raise ValueError(method)
        intended, length = length, min(length, budget - used)
        runs.append((pop, length, intended))
        used += length
        r += 1
        if method == 'CMA-ES':
            break
    return runs


def schedules_that_fit(interval):
    return [interval * (2 ** k - 1) for k in (2, 3, 4)]


def schedule_is_clipped(method, budget, interval):
    return any(length < intended for _, length, intended in run_schedule(method, budget, interval))


def cma_run(a, d, budget, seed, method, restart_interval):
    import cma
    rec = Recorder(a, d)
    x_first = np.random.default_rng(seed).uniform(LO, HI, size=2)
    restart_rng = np.random.default_rng(np.random.SeedSequence([seed, 424242]))
    populations, segments, generations, clipped = [], [], [], False
    for r, (pop, length, intended) in enumerate(run_schedule(method, budget, restart_interval)):
        x0 = x_first if r == 0 else restart_rng.uniform(LO, HI, size=2)
        cma_seed = seed + 1 if r == 0 else int(restart_rng.integers(1, 2 ** 31 - 1))
        opts = {'bounds': [[LO, LO], [HI, HI]], 'seed': cma_seed, 'verbose': -9, 'maxfevals': length}
        if r > 0:
            opts['popsize'] = pop
        es = cma.CMAEvolutionStrategy(x0, 2.0, opts)
        if r == 0 and int(es.popsize) != BASE_POP:
            raise RuntimeError(f'default population is {es.popsize}, expected {BASE_POP}')
        populations.append(int(es.popsize))
        start = len(rec.points)
        limit = start + length
        gens = 0
        while not es.stop() and len(rec.points) + es.popsize <= limit:
            xs = es.ask()
            ys = [rec.add(x) for x in xs]
            es.tell(xs, ys)
            gens += 1
        remaining = limit - len(rec.points)
        if remaining:
            for x in es.ask(number=remaining):
                rec.add(x)
        segments.append(len(rec.points) - start)
        generations.append(gens)
        clipped |= length < intended
    out = rec.finish()
    out.update(restart_count=len(populations) - 1, population_sizes=populations,
               segment_evaluations=segments, generations_per_run=generations,
               total_generations=int(sum(generations)), schedule_clipped=bool(clipped))
    return out


def reference_run(a, d, budget, seed, method):
    rec = Recorder(a, d)
    if method == 'Random':
        unit = np.random.default_rng(seed).random((budget, 2))
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            unit = qmc.Sobol(2, scramble=True, seed=seed).random(budget)
    for u in unit:
        rec.add(LO + (HI - LO) * u)
    out = rec.finish()
    out.update(restart_count=0, population_sizes=[], generations_per_run=[], total_generations=None,
               schedule_clipped=False)
    return out


def run_one(job):
    a, d, method, seed, budget, folder, restart_interval = job
    path = Path(folder) / 'runs' / f'a{a:g}_D{d:g}_{method}_s{seed}.json'
    if path.exists():
        return str(path), 'skipped'
    start = time.perf_counter()
    r = (cma_run(a, d, budget, seed, method, restart_interval) if method.startswith('CMA-ES')
         else reference_run(a, d, budget, seed, method))
    r.update(alpha=a, distance=d, cell=f'a{a:g}_D{d:g}', acquisition=method,
             seed=seed, budget=budget, runtime_s=time.perf_counter() - start)
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


LISTS = ('history', 'population_sizes', 'segment_evaluations', 'generations_per_run')


def aggregate(folder):
    folder = Path(folder)
    recs = [json.loads(p.read_text()) for p in sorted((folder / 'runs').glob('*.json'))]
    if not recs:
        raise RuntimeError('No saved runs found')
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in LISTS} for r in recs])
    if 'total_generations' not in df:
        df['total_generations'] = np.nan
    df['total_generations'] = pd.to_numeric(df['total_generations'], errors='coerce')
    if 'schedule_clipped' not in df:
        df['schedule_clipped'] = False
    df['schedule_clipped'] = df['schedule_clipped'].fillna(False).astype(bool)
    df.to_csv(folder / 'individual_runs.csv', index=False)

    summary = []
    for (cell, method), g in df.groupby(['cell', 'acquisition']):
        row = dict(cell=cell, alpha=g.alpha.iloc[0], distance=g.distance.iloc[0],
                   acquisition=method, n=len(g))
        for metric in ('deep', 'success', 'branch'):
            k = int(g[metric].sum())
            low, high = wilson(k, len(g))
            row.update({f'{metric}_k': k, f'{metric}_rate': k / len(g),
                        f'{metric}_lo': low, f'{metric}_hi': high})
        row.update(mean_final_regret=g.final_regret.mean(),
                   median_final_regret=g.final_regret.median(),
                   mean_restart_count=g.restart_count.mean(),
                   mean_total_generations=g.total_generations.mean(),
                   min_evals_used=int(g.evaluations_used.min()),
                   median_evals_used=g.evaluations_used.median(),
                   median_runtime_s=g.runtime_s.median(),
                   n_schedule_clipped=int(g.schedule_clipped.sum()),
                   frac_schedule_clipped=float(g.schedule_clipped.mean()))
        summary.append(row)
    summary = pd.DataFrame(summary).sort_values(['alpha', 'distance', 'acquisition'])
    summary.to_csv(folder / 'summary.csv', index=False)
    print(summary.to_string(index=False))

    comparisons = [
        ('CMA-ES', 'CMA-ES-RR', 'primary'),
        ('CMA-ES', 'CMA-ES-IPOP', 'primary'),
        ('CMA-ES', 'CMA-ES-IPOP-G', 'primary'),
        ('CMA-ES', 'CMA-ES-RR-S', 'primary'),
        ('CMA-ES-RR', 'CMA-ES-IPOP', 'ladder'),
        ('CMA-ES-RR', 'CMA-ES-IPOP', 'population'),
        ('CMA-ES-RR-S', 'CMA-ES-IPOP-G', 'population'),
        ('CMA-ES-RR-S', 'CMA-ES-RR', 'population'),
        ('CMA-ES-IPOP', 'CMA-ES-IPOP-G', 'schedule'),
        ('CMA-ES', 'Random', 'reference'),
        ('CMA-ES', 'Sobol', 'reference'),
    ]
    tests = []
    for metric in ('deep', 'success', 'branch'):
        group = []
        for cell, g in df.groupby('cell'):
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
        for family in ('primary', 'ladder', 'population', 'schedule', 'reference'):
            subset = [x for x in group if x['family'] == family]
            for x, adjusted in zip(subset, holm([x['p'] for x in subset])):
                x['holm_p'] = adjusted
            tests.extend(subset)
    pd.DataFrame(tests).to_csv(folder / 'paired_tests.csv', index=False)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for metric in ('deep', 'success'):
        fig, ax = plt.subplots(figsize=(11, 4))
        cells = list(summary.cell.unique())
        methods = [m for m in METHODS if m in set(summary.acquisition)]
        width = .8 / len(methods)
        for j, method in enumerate(methods):
            s = summary[summary.acquisition == method].set_index('cell').reindex(cells)
            ax.bar(np.arange(len(cells)) + (j - (len(methods) - 1) / 2) * width,
                   s[f'{metric}_rate'], width, label=method)
        ax.set_xticks(np.arange(len(cells)), cells)
        ax.set_ylim(0, 1)
        ax.set_ylabel(f'{metric} rate')
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(folder / f'{metric}_rates.png', dpi=180)
        plt.close(fig)
    print('Saved outputs in', folder)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', nargs='+', default=['5:4', '10:4', '20:4', '10:1', '10:2', '10:3'])
    ap.add_argument('--acquisitions', nargs='+', default=list(METHODS), choices=METHODS)
    ap.add_argument('--runs', type=int, default=100)
    ap.add_argument('--seed-start', type=int, default=3000)
    ap.add_argument('--budget', type=int, default=420)
    ap.add_argument('--restart-interval', type=int, default=60)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--output', type=Path, default=None,
                    help='defaults to results/experiment4_schedule<interval>_b<budget>')
    ap.add_argument('--allow-clipped', action='store_true',
                    help='permit a budget that clips IPOP-G / RR-S (the generation-matched contrast is then unclean)')
    ap.add_argument('--aggregate-only', action='store_true')
    args = ap.parse_args()

    if args.budget < 1 or args.runs < 1 or args.workers < 1:
        ap.error('budget, runs, and workers must be positive')
    if args.restart_interval < 1:
        ap.error('--restart-interval must be positive')
    if args.output is None:
        args.output = Path(f'results/experiment4_schedule{args.restart_interval}_b{args.budget}')
    if set(args.acquisitions) & set(G_ARMS) and args.restart_interval % BASE_POP:
        ap.error(f'equal-generation arms need --restart-interval divisible by {BASE_POP} '
                 f'(e.g. 60 or 120)')
    if not args.aggregate_only and set(args.acquisitions) & set(G_ARMS):
        clipped = [m for m in G_ARMS if m in args.acquisitions
                   and schedule_is_clipped(m, args.budget, args.restart_interval)]
        if clipped:
            fits = schedules_that_fit(args.restart_interval)
            msg = (f'budget {args.budget} clips the last run of {", ".join(clipped)}: the final run is not '
                   f'generation-matched, so the population family is not a clean contrast. The schedule fits '
                   f'exactly at budgets {fits}.')
            if not args.allow_clipped:
                ap.error(msg + ' Re-run with --allow-clipped to proceed anyway, or use one of the listed budgets.')
            print('WARNING: ' + msg, flush=True)

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
        fixed = dict(budget=args.budget, sigma0=2.0, setup='original_exp1', restart_trigger='scheduled',
                     restart_interval=args.restart_interval)
        config_file = args.output / 'config_used.json'
        invocations = []
        if config_file.exists():
            old = json.loads(config_file.read_text())
            clash = {k: (old[k], fixed[k]) for k in fixed if k in old and old[k] != fixed[k]}
            if clash:
                ap.error(f'Output directory has different settings {clash}; choose a new --output path')
            invocations = old.get('invocations') or (
                [dict(cells=old['cells'], acquisitions=old['acquisitions'], seed_start=old['seed_start'],
                      runs=old['runs'])] if 'cells' in old else [])
        invocations.append(dict(cells=args.cells, acquisitions=args.acquisitions,
                                seed_start=args.seed_start, runs=args.runs))
        config_file.write_text(json.dumps(dict(fixed, invocations=invocations), indent=2))
        jobs = [(a, d, m, args.seed_start + i, args.budget, str(args.output), args.restart_interval)
                for a, d in cells for m in args.acquisitions for i in range(args.runs)]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_one, j) for j in jobs]
            for i, future in enumerate(as_completed(futures), 1):
                path, status = future.result()
                print(f'[{i}/{len(jobs)}] {Path(path).name}: {status}', flush=True)
    aggregate(args.output)


if __name__ == '__main__':
    main()