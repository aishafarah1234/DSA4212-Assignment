"""Matched-initialisation control: CMA-ES given the same first n_initial evaluations as BO.

BO starts from n_initial scrambled Sobol points (n_initial = 10 in the pilot setup, 8 in the aligned setup).
Here CMA-ES gets exactly that design (same torch SobolEngine, same seed, same scaling), counted inside the
budget, then starts from the best of those points with sigma0 = 2 and the default population for the
remaining evaluations. Because the seeds and the first n_initial points are shared, the comparison with the
saved BO (LogEI) runs is genuinely paired -- provided point-by-point verification passes. Unverified pairs
are excluded from the analysis.

"Matched" means matched EVALUATION BUDGET, not matched information: BO's GP sees all n_initial points;
CMA-ES's mean sees only the argmin. That is the natural way to hand CMA-ES a design, and it is the choice
this script makes.

Pairing labels in paired_tests.csv:
  'genuine'  = same first n_initial evaluations are shared (matched family).
  'nominal'  = same seed index only (unmatched and init families).
The init family (CMA-ES vs CMA-ES-S10) is NOT genuinely paired: CMA-ES starts from a random mean, S10 from
the argmin of the shared design.

  python code/matched_init.py --source results/experiment3 --cells 5:4 10:4 20:4 10:1 --seed-start 2000 --runs 30

Arms: CMA-ES (usual random start, for reference), CMA-ES-S10 (shared design, then CMA-ES).
Tests (exact McNemar, Holm within metric and family): S10 vs LogEI ("matched"), CMA-ES vs LogEI
("unmatched"), S10 vs CMA-ES ("init", nominal).
Uses Recorder, cma_run, wilson and holm from experiment4.py; the e4 contract is asserted at startup.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

import experiment4 as e4

LO, HI = e4.LO, e4.HI
SHARED_ACQS = ('LogEI',)


def assert_e4_contract():
    required = ('Recorder', 'cma_run', 'wilson', 'holm', 'LO', 'HI', 'BASE_POP', 'METHODS')
    missing = [name for name in required if not hasattr(e4, name)]
    if missing:
        raise SystemExit(f'experiment4.py is missing {missing}; update matched_init.py or restore the expected API')
    if e4.BASE_POP != 6:
        raise SystemExit(f'experiment4.BASE_POP is {e4.BASE_POP}, expected 6')
    if 'CMA-ES' not in e4.METHODS:
        raise SystemExit("experiment4.METHODS does not contain 'CMA-ES'")
    import inspect
    params = list(inspect.signature(e4.cma_run).parameters)
    expected = ['a', 'd', 'budget', 'seed', 'method', 'restart_interval']
    if params != expected:
        raise SystemExit(f'e4.cma_run signature is {params}, expected {expected}')


def e4_fingerprint():
    src = Path(e4.__file__)
    return dict(path=str(src), sha256=hashlib.sha256(src.read_bytes()).hexdigest()[:16])


def read_source_n_initial(source):
    cfg_path = source / 'config_used.json'
    if not cfg_path.exists():
        raise SystemExit(f'{cfg_path} not found: cannot determine the BO arm\'s n_initial. '
                         f'Pass --n-init explicitly if you know it.')
    cfg = json.loads(cfg_path.read_text())
    n = cfg.get('fixed', {}).get('n_initial')
    if n is None:
        raise SystemExit(f'{cfg_path} has no fixed.n_initial; pass --n-init explicitly.')
    return int(n)


def validate_source(bo, cells, seed_start, runs):
    wanted = {(f'a{a:g}_D{d:g}', s) for (a, d), s in
              ((tuple(map(float, c.split(':'))), s) for c in cells for s in range(seed_start, seed_start + runs))}
    counts = bo.groupby(['cell', 'seed']).size()
    missing, duplicated = [], []
    for cell, seed in sorted(wanted):
        n = int(counts.get((cell, seed), 0))
        if n == 0:
            missing.append((cell, seed))
        elif n > 1:
            duplicated.append((cell, seed, n))
    if missing or duplicated:
        msg = []
        if missing:
            msg.append(f'{len(missing)} missing (cell, seed) pairs, e.g. {missing[:5]}')
        if duplicated:
            msg.append(f'{len(duplicated)} duplicated (cell, seed) pairs, e.g. {duplicated[:5]}')
        raise SystemExit('source validation failed: ' + '; '.join(msg))
    return bo


def shared_design(seed, n):
    from torch.quasirandom import SobolEngine
    u = SobolEngine(2, scramble=True, seed=seed).draw(n).double().numpy()
    return LO + (HI - LO) * u


def cma_s10(a, d, budget, seed, n_init):
    import cma
    rec = e4.Recorder(a, d)
    design = shared_design(seed, n_init)
    ys = [rec.add(x) for x in design]
    es = cma.CMAEvolutionStrategy(design[int(np.argmin(ys))], 2.0,
                                  {'bounds': [[LO, LO], [HI, HI]], 'seed': seed + 1,
                                   'verbose': -9, 'maxfevals': budget - n_init})
    while not es.stop() and len(rec.points) + es.popsize <= budget:
        xs = es.ask()
        es.tell(xs, [rec.add(x) for x in xs])
    left = budget - len(rec.points)
    if left:
        for x in es.ask(number=left):
            rec.add(x)
    out = rec.finish()
    out['design'] = design.tolist()
    out['n_init'] = n_init
    return out


def mcnemar(base, alt):
    b, c = int(np.sum(~base & alt)), int(np.sum(base & ~alt))
    return (binomtest(b, b + c, .5).pvalue if b + c else 1.), b, c


def first_mismatch(pts, design):
    """Index of the first point where pts and design disagree, or -1 if they agree everywhere.
    Returns min(len(pts), len(design)) if the arrays have different lengths, so a truncated trajectory
    (e.g. a saved BO run with fewer than n_init points) fails verification rather than being accepted
    on the shared prefix."""
    pts = np.asarray(pts, dtype=float)
    design = np.asarray(design, dtype=float)
    if len(pts) != len(design):
        return min(len(pts), len(design))
    if len(pts) == 0:
        return 0
    diff = np.any(np.abs(pts - design) > 1e-9, axis=1)
    return int(np.argmax(diff)) if diff.any() else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, default=Path('results/experiment3'),
                    help='runner folder holding the BO (LogEI) runs')
    ap.add_argument('--cells', nargs='+', default=['5:4', '10:4', '20:4', '10:1'])
    ap.add_argument('--seed-start', type=int, default=2000)
    ap.add_argument('--runs', type=int, default=30)
    ap.add_argument('--budget', type=int, default=300)
    ap.add_argument('--n-init', type=int, default=None,
                    help='number of shared initial points; defaults to the source\'s fixed.n_initial')
    ap.add_argument('--output', type=Path, default=Path('results/matched_init'))
    ap.add_argument('--max-unverified-fraction', type=float, default=0.0,
                    help='fraction of pairs that may fail point-by-point verification before SystemExit '
                         '(default 0: every pair must verify)')
    args = ap.parse_args()

    assert_e4_contract()

    src_n_init = read_source_n_initial(args.source)
    n_init = src_n_init if args.n_init is None else args.n_init
    if n_init != src_n_init:
        raise SystemExit(f'source uses n_initial={src_n_init}, but --n-init={n_init}; '
                         f'the "matched" family would not be matched. Re-run with --n-init {src_n_init}.')
    print(f'using n_init={n_init} (matches {args.source}/config_used.json)')

    args.output.mkdir(parents=True, exist_ok=True)
    bo = pd.read_csv(args.source / 'individual_runs.csv')
    bo = bo[bo.acquisition.isin(SHARED_ACQS)]
    if bo.empty:
        raise SystemExit(f'no {SHARED_ACQS} rows in {args.source}/individual_runs.csv')
    bo = validate_source(bo, args.cells, args.seed_start, args.runs)
    print(f'source validated: one LogEI row per requested (cell, seed), {len(bo)} rows')

    rows, verified, checked, mismatches = [], 0, 0, []
    for cell in args.cells:
        a, d = map(float, cell.split(':'))
        label = f'a{a:g}_D{d:g}'
        for seed in range(args.seed_start, args.seed_start + args.runs):
            runs = {
                'CMA-ES': e4.cma_run(a, d, args.budget, seed, 'CMA-ES', 60),
                'CMA-ES-S10': cma_s10(a, d, args.budget, seed, n_init),
            }
            f = args.source / 'runs' / f'{label}_LogEI_s{seed}.json'
            if not f.exists():
                raise SystemExit(f'{f} not found; source validation passed for individual_runs.csv '
                                 f'but the JSON with the first points is missing')
            pts = json.loads(f.read_text()).get('points')
            if not pts:
                raise SystemExit(f'{f} has no saved points; cannot verify the shared design')
            checked += 1
            idx = first_mismatch(pts[:n_init], runs['CMA-ES-S10']['design'][:n_init])
            if idx < 0:
                verified += 1
            else:
                mismatches.append((label, seed, idx))
                continue
            for arm, r in runs.items():
                rows.append(dict(cell=label, alpha=a, distance=d, acquisition=arm, seed=seed,
                                 deep=r['deep'], success=r['success'], branch=r['branch'],
                                 final_regret=r['final_regret'], evaluations_used=r['evaluations_used']))

    if checked == 0:
        raise SystemExit('no pairs checked; cannot run a matched analysis')
    frac = len(mismatches) / checked
    print(f'initial design identical in {verified} of {checked} pairs')
    if mismatches:
        print('first mismatches (cell, seed, index):', mismatches[:5])
    if frac > args.max_unverified_fraction:
        raise SystemExit(f'{len(mismatches)} of {checked} pairs ({frac:.1%}) failed verification, '
                         f'above --max-unverified-fraction={args.max_unverified_fraction:.1%}. '
                         f'The matched family would not be matched; fix the source or pass a higher threshold.')

    mine = pd.DataFrame(rows)
    if mine.empty:
        raise SystemExit('no verified pairs; nothing to analyse')

    verified_pairs = mine[['cell', 'seed']].drop_duplicates()
    bo_keep = bo.merge(verified_pairs, on=['cell', 'seed'], how='inner')[
        ['cell', 'alpha', 'distance', 'acquisition', 'seed', 'deep', 'success', 'branch',
         'final_regret', 'evaluations_used']]
    if len(bo_keep) != len(verified_pairs):
        raise SystemExit(f'expected {len(verified_pairs)} verified BO rows, got {len(bo_keep)}; '
                         f'validate_source should have guaranteed one LogEI row per verified (cell, seed)')

    allr = pd.concat([mine, bo_keep], ignore_index=True)
    allr['pairing'] = np.where(allr.acquisition.isin(['LogEI', 'CMA-ES-S10']), 'genuine', 'nominal')
    allr.to_csv(args.output / 'individual_runs.csv', index=False)

    summ = []
    for (cell, arm), g in allr.groupby(['cell', 'acquisition']):
        row = dict(cell=cell, acquisition=arm, pairing=g.pairing.iloc[0], n=len(g))
        for m in ('deep', 'success'):
            k = int(g[m].sum())
            lo, hi = e4.wilson(k, len(g))
            row.update({f'{m}_k': k, f'{m}_rate': k / len(g), f'{m}_lo': lo, f'{m}_hi': hi})
        summ.append(row)
    pd.DataFrame(summ).to_csv(args.output / 'summary.csv', index=False)

    comps = [
        ('LogEI',  'CMA-ES-S10', 'matched',   'genuine'),
        ('LogEI',  'CMA-ES',     'unmatched', 'nominal'),
        ('CMA-ES', 'CMA-ES-S10', 'init',      'nominal'),
    ]
    tests = []
    for m in ('deep', 'success'):
        group = []
        for cell, g in allr.groupby('cell'):
            for b_arm, a_arm, fam, pairing in comps:
                b = g[g.acquisition == b_arm].set_index('seed')[m].astype(bool)
                x = g[g.acquisition == a_arm].set_index('seed')[m].astype(bool)
                s = b.index.intersection(x.index)
                if len(s):
                    p, alt_only, base_only = mcnemar(b[s].to_numpy(), x[s].to_numpy())
                    group.append(dict(metric=m, family=fam, pairing=pairing, cell=cell,
                                      baseline=b_arm, alternative=a_arm, n_pairs=len(s),
                                      baseline_rate=b[s].mean(), alt_rate=x[s].mean(),
                                      alt_only=alt_only, baseline_only=base_only, p=p))
        for fam in ('matched', 'unmatched', 'init'):
            sub = [t for t in group if t['family'] == fam]
            for t, h in zip(sub, e4.holm([t['p'] for t in sub])):
                t['holm_p'] = h
            tests += sub
    pd.DataFrame(tests).to_csv(args.output / 'paired_tests.csv', index=False)

    cfg = dict(source=str(args.source), n_init=n_init, budget=args.budget, cells=args.cells,
               seed_start=args.seed_start, runs=args.runs, verified=verified, checked=checked,
               unverified=[list(map(str, mm)) for mm in mismatches], experiment4=e4_fingerprint())
    (args.output / 'config_used.json').write_text(json.dumps(cfg, indent=2))
    print(pd.DataFrame(summ).round(3).to_string(index=False))
    print(pd.DataFrame(tests).round(4).to_string(index=False))
    print('Saved to', args.output)


if __name__ == '__main__':
    main()