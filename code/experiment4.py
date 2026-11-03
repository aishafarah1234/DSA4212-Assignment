"""Standalone Experiment 4: scheduled CMA-ES restarts.

Uses objective.two_basin; no changes to runner.py are needed.
Run: python code/experiment4.py --workers 3
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from math import sqrt

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from scipy.stats import qmc
from objective import two_basin

LO, HI, DELTA, TOL = -5., 5., 1., .01
METHODS = ('CMA-ES', 'CMA-ES-RR', 'CMA-ES-IPOP', 'Random', 'Sobol')


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


def cma_run(a, d, budget, seed, method, restart_interval):
    """Match original Exp 1 baseline; scheduled variants share its first segment.

    At each scheduled boundary, finish only complete CMA generations, then
    evaluate any residual segment slots without tell(). This ensures exact
    budgets but the final partial generation is not used for adaptation.
    """
    import cma
    rec = Recorder(a, d)
    x_first = np.random.default_rng(seed).uniform(LO, HI, size=2)
    restart_rng = np.random.default_rng(np.random.SeedSequence([seed, 424242]))
    restart_count = 0
    populations = []
    segments = []
    base_pop = None
    while len(rec.points) < budget:
        first = len(populations) == 0
        x0 = x_first if first else restart_rng.uniform(LO, HI, size=2)
        cma_seed = seed + 1 if first else int(restart_rng.integers(1, 2**31 - 1))
        opts = {'bounds': [[LO, LO], [HI, HI]], 'seed': cma_seed,
                'verbose': -9, 'maxfevals': budget}
        if base_pop is not None:
            opts['popsize'] = base_pop * (2 ** restart_count) if method == 'CMA-ES-IPOP' else base_pop
        es = cma.CMAEvolutionStrategy(x0, 2.0, opts)
        if base_pop is None:
            base_pop = int(es.popsize)
        populations.append(int(es.popsize))
        start = len(rec.points)
        limit = budget if method == 'CMA-ES' else min(budget, start + restart_interval)
        while not es.stop() and len(rec.points) + es.popsize <= limit:
            xs = es.ask()
            ys = [rec.add(x) for x in xs]
            es.tell(xs, ys)
        remaining = limit - len(rec.points)
        if remaining:
            for x in es.ask(number=remaining):
                rec.add(x)
        segments.append(len(rec.points) - start)
        if len(rec.points) >= budget or method == 'CMA-ES':
            break
        restart_count += 1
    out = rec.finish()
    out.update(restart_count=restart_count, population_sizes=populations,
               segment_evaluations=segments)
    return out


def reference_run(a, d, budget, seed, method):
    rec = Recorder(a, d)
    if method == 'Random':
        unit = np.random.default_rng(seed).random((budget, 2))
    else:
        with __import__('warnings').catch_warnings():
            __import__('warnings').simplefilter('ignore')
            unit = qmc.Sobol(2, scramble=True, seed=seed).random(budget)
    for u in unit:
        rec.add(LO + (HI - LO) * u)
    out = rec.finish()
    out.update(restart_count=0, population_sizes=[])
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
    den = 1 + z*z/n
    mid = (p + z*z/(2*n))/den
    rad = z * sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return max(0., mid-rad), min(1., mid+rad)


def holm(pvalues):
    order = np.argsort(pvalues)
    out = [0.] * len(pvalues)
    prev = 0.
    for rank, ix in enumerate(order):
        prev = max(prev, (len(pvalues)-rank)*pvalues[ix])
        out[ix] = min(1., prev)
    return out


def aggregate(folder):
    folder = Path(folder)
    recs = [json.loads(p.read_text()) for p in sorted((folder/'runs').glob('*.json'))]
    if not recs:
        raise RuntimeError('No saved runs found')
    df = pd.DataFrame([{k:v for k,v in r.items() if k not in ('history','population_sizes','segment_evaluations')} for r in recs])
    df.to_csv(folder/'individual_runs.csv', index=False)
    summary = []
    for (cell, method), g in df.groupby(['cell','acquisition']):
        row = dict(cell=cell, alpha=g.alpha.iloc[0], distance=g.distance.iloc[0],
                   acquisition=method, n=len(g))
        for metric in ('deep','success','branch'):
            k = int(g[metric].sum())
            low, high = wilson(k, len(g))
            row.update({f'{metric}_k':k, f'{metric}_rate':k/len(g),
                        f'{metric}_lo':low, f'{metric}_hi':high})
        row.update(mean_final_regret=g.final_regret.mean(),
                   median_final_regret=g.final_regret.median(),
                   mean_restart_count=g.restart_count.mean(),
                   min_evals_used=int(g.evaluations_used.min()),
                   median_evals_used=g.evaluations_used.median(),
                   median_runtime_s=g.runtime_s.median())
        summary.append(row)
    summary = pd.DataFrame(summary).sort_values(['alpha','distance','acquisition'])
    summary.to_csv(folder/'summary.csv', index=False)
    print(summary.to_string(index=False))
    tests = []
    # Primary: RR vs CMA, IPOP vs CMA; secondary: IPOP vs RR;
    # references: Random/Sobol vs CMA. Holm separately per metric and family.
    comparisons = [('CMA-ES','CMA-ES-RR','primary'),
                   ('CMA-ES','CMA-ES-IPOP','primary'),
                   ('CMA-ES-RR','CMA-ES-IPOP','ladder'),
                   ('CMA-ES','Random','reference'),
                   ('CMA-ES','Sobol','reference')]
    for metric in ('deep','success','branch'):
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
        for family in ('primary','ladder','reference'):
            subset = [x for x in group if x['family'] == family]
            for x, adjusted in zip(subset, holm([x['p'] for x in subset])):
                x['holm_p'] = adjusted
            tests.extend(subset)
    pd.DataFrame(tests).to_csv(folder/'paired_tests.csv', index=False)
    # Save a simple plot without introducing another module.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for metric in ('deep','success'):
        fig, ax = plt.subplots(figsize=(10,4))
        cells = list(summary.cell.unique())
        methods = [m for m in METHODS if m in set(summary.acquisition)]
        width = .8/len(methods)
        for j, method in enumerate(methods):
            s = summary[summary.acquisition == method].set_index('cell').reindex(cells)
            ax.bar(np.arange(len(cells)) + (j-(len(methods)-1)/2)*width,
                   s[f'{metric}_rate'], width, label=method)
        ax.set_xticks(np.arange(len(cells)), cells)
        ax.set_ylim(0,1)
        ax.set_ylabel(f'{metric} rate')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(folder/f'{metric}_rates.png', dpi=180)
        plt.close(fig)
    print('Saved outputs in', folder)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', nargs='+', default=['5:4','10:4','20:4','10:1','10:2','10:3'])
    ap.add_argument('--acquisitions', nargs='+', default=list(METHODS), choices=METHODS)
    ap.add_argument('--runs', type=int, default=100)
    ap.add_argument('--seed-start', type=int, default=3000)
    ap.add_argument('--budget', type=int, default=300)
    ap.add_argument('--restart-interval', type=int, default=60)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--output', type=Path, default=Path('results/experiment4'))
    ap.add_argument('--aggregate-only', action='store_true')
    args = ap.parse_args()
    if args.budget < 1 or args.runs < 1 or args.workers < 1:
        ap.error('budget, runs, and workers must be positive')
    if args.restart_interval < 1:
        ap.error('--restart-interval must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'runs').mkdir(exist_ok=True)
    if not args.aggregate_only:
        cells = []
        for s in args.cells:
            a,d = map(float,s.split(':'))
            radius = sqrt(DELTA/a)
            if not (LO < d-2-radius and d-2+radius < HI):
                ap.error(f'Global negative-value region outside box: {s}')
            cells.append((a,d))
        config = dict(budget=args.budget, cells=args.cells, acquisitions=args.acquisitions,
                      seed_start=args.seed_start, runs=args.runs,
                      sigma0=2.0, setup='original_exp1', restart_trigger='scheduled',
                      restart_interval=args.restart_interval)
        config_file = args.output/'config_used.json'
        if config_file.exists() and json.loads(config_file.read_text()) != config:
            ap.error('Output directory has different settings; choose a new --output path')
        config_file.write_text(json.dumps(config,indent=2))
        jobs = [(a,d,m,args.seed_start+i,args.budget,str(args.output),args.restart_interval)
                for a,d in cells for m in args.acquisitions for i in range(args.runs)]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_one,j) for j in jobs]
            for i, future in enumerate(as_completed(futures),1):
                path,status = future.result()
                print(f'[{i}/{len(jobs)}] {Path(path).name}: {status}',flush=True)
    aggregate(args.output)


if __name__ == '__main__':
    main()
