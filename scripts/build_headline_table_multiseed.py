"""Render the multi-seed headline table (docs/table_multiseed_ne<N>[_moe]_<model>.csv).

Same cells as build_headline_table.py, but aggregated over several training seeds:
each (dataset, method) carries coverage/width/mse/mae as mean + sample std across the
seeds that actually have a run, plus the count so a partially-filled cell is visible
rather than silently averaged over fewer seeds.

    python scripts/build_headline_table_multiseed.py --num-experts 3 --variant MOG \
        --seeds 4021,4022,4023,4024,4025

Deliberately a separate script rather than a flag on build_headline_table.py: that one
feeds the base (single-seed) tables in docs/results_tables.tex, and widening its schema
in place would put those at risk for no benefit. The method list, the n/a rules and the
CSV location are imported from it, so the two cannot drift on what a cell *means* --
only on how many runs go into it.

std is the sample std (ddof=1) and is empty when only one seed is present, which is the
honest rendering: one run has no spread, and 0.0000 would imply it was measured to be
zero. Downstream (build_results_tex.py) prints mean alone in that case.
"""
import argparse
import csv
import os
import statistics

from build_headline_table import CSV, DATASETS, METHODS, NA, NEEDS_PROB, PL, ROOT


def agg(vals):
    """(mean, sample-std, n) over the non-empty values; std is None below 2 runs."""
    xs = [float(v) for v in vals if v not in (None, '', NA)]
    if not xs:
        return None, None, 0
    return (statistics.fmean(xs),
            statistics.stdev(xs) if len(xs) > 1 else None,
            len(xs))


def fmt(x, nd=4):
    return '' if x is None else f'{x:.{nd}f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num-experts', default='3')
    ap.add_argument('--variant', default='MOG', choices=['MOG', 'MOE'])
    ap.add_argument('--seeds', default='4021,4022,4023,4024,4025')
    ap.add_argument('--model', default='iTransformer')
    ap.add_argument('--out')
    args = ap.parse_args()

    seeds = [s.strip() for s in args.seeds.split(',') if s.strip()]
    tag = '' if args.variant == 'MOG' else '_moe'
    out = args.out or os.path.join(
        ROOT, 'docs',
        f'table_multiseed_ne{args.num_experts}{tag}_{args.model.lower()}.csv')

    # (dataset, pred_len, method, variant, model_id, seed) -> row, i.e. the single-seed
    # key of build_headline_table.py with the seed promoted into the key instead of a filter.
    cells = {}
    with open(CSV) as fh:
        for r in csv.DictReader(fh):
            if (r['model'] == args.model and r['num_experts'] == args.num_experts
                    and r['seed'] in seeds and r['features'] == 'M'
                    and r['unc_gating'] == '0' and r['coverage']):
                cells[(r['dataset'], int(r['pred_len']), r['method'],
                       r['variant'], r['model_id'], r['seed'])] = r

    header = ['dataset', 'pred_len', 'n_seeds_requested',
              'mse_mean', 'mse_std', 'mae_mean', 'mae_std', 'err_n_seeds']
    for label, *_ in METHODS:
        header += [f'{label}_coverage_mean', f'{label}_coverage_std',
                   f'{label}_width_mean', f'{label}_width_std', f'{label}_n_seeds',
                   f'{label}_mse_mean', f'{label}_mse_std',
                   f'{label}_mae_mean', f'{label}_mae_std']

    rows = []
    for d in DATASETS:
        pl = PL.get(d, 96)
        row = [d, pl, len(seeds)]

        # Block-level mse/mae: the requested variant's own backbone, as in the single-seed
        # builder. Kept for the model blocks whose rows all share one model.
        errs = [cells.get((d, pl, 'standard_cp', args.variant, 'test', s)) for s in seeds]
        mse_m, mse_s, _ = agg([e['mse'] for e in errs if e])
        mae_m, mae_s, n_err = agg([e['mae'] for e in errs if e])
        row += [fmt(mse_m, 6), fmt(mse_s, 6), fmt(mae_m, 6), fmt(mae_s, 6), n_err]

        for _, slug, model_id in METHODS:
            if slug in NEEDS_PROB and args.variant == 'MOE':
                row += [NA] * 9
                continue
            variant = 'MOE' if model_id == 'cqr' else args.variant
            got = [cells.get((d, pl, slug, variant, model_id, s)) for s in seeds]
            got = [g for g in got if g]
            cov_m, cov_s, n = agg([g['coverage'] for g in got])
            wid_m, wid_s, _ = agg([g['width'] for g in got])
            # Per-method mse/mae, read off each method's *own* rows rather than the block's.
            # This matters for the two CQR columns: they run on the separate pinball-loss
            # trunk (model_id=cqr), so the block backbone's error is not theirs, and pinning
            # the block value on them would misreport a different model's accuracy.
            m_mse_m, m_mse_s, _ = agg([g['mse'] for g in got])
            m_mae_m, m_mae_s, _ = agg([g['mae'] for g in got])
            row += [fmt(cov_m), fmt(cov_s), fmt(wid_m), fmt(wid_s), n,
                    fmt(m_mse_m, 6), fmt(m_mse_s, 6), fmt(m_mae_m, 6), fmt(m_mae_s, 6)]
        rows.append(row)

    with open(out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)

    print(f'wrote {out}  (seeds requested: {", ".join(seeds)})')
    for row in rows:
        # n_seeds for each method sits at 8 + 9*i + 4 in the row layout above.
        partial = []
        for i, (label, *_) in enumerate(METHODS):
            n = row[8 + 9 * i + 4]
            if n == NA:
                continue
            if n == 0:
                partial.append(f'{label}:0')
            elif n < len(seeds):
                partial.append(f'{label}:{n}/{len(seeds)}')
        if partial:
            print(f'  PARTIAL {row[0]}: {", ".join(partial)}')


if __name__ == '__main__':
    main()
