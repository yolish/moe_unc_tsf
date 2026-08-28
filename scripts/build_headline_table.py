"""Render the headline calibration table (docs/table_seed<seed>_ne<N>_<model>.csv).

Reads docs/calibration_results_tsf.csv (regenerate it first with
collect_calibration_results.py). One row per dataset at the headline horizon
(pred_len 96, national-illness 24), one coverage/width column pair per method.

    python scripts/build_headline_table.py --num-experts 1

Empty cells mean the run is missing. Cells reading "n/a" are impossible for the
requested variant, not missing -- see NEEDS_PROB below.
"""
import argparse
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, 'docs', 'calibration_results_tsf.csv')

DATASETS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'weather', 'electricity',
            'traffic', 'exchange-rate', 'national-illness']
PL = {'national-illness': 24}

# (column prefix, collector method slug, model_id). The variant is the one being built,
# except for the two CQR columns: CQR is always its own pinball-loss model (model_id=cqr,
# no --prob_expert), so those cells are the same numbers whichever variant is requested.
# Each ACI variant sits immediately after its base so the table reads base/ACI adjacent.
# All the ACI_g0.001 columns share --aci_gamma 0.001 (the 0.1/pred_len rule of thumb at
# pred_len=96); see scripts/run_aci_base_methods.sh.
METHODS = [
    ('CP',                  'standard_cp',              'test'),
    ('CP_ACI_g0.001',       'aci_cp',                   'test'),
    ('CPVS',                'cpvs',                     'test'),
    ('CPVS_ACI_g0.001',     'aci_cpvs',                 'test'),
    ('CPVS_aleatoric',      'aleatoric_only',           'test'),
    ('CPVS_aleatoric_ACI_g0.001', 'aci_aleatoric_only', 'test'),
    ('CP_MOG',              'cp_aleatoric_scale',       'test'),
    ('CP_MOG_ACI_g0.01',    'aci_aleatoric_scale',      'test'),
    ('CP_MOG_ACI_g0.001',   'aci_aleatoric_scale_g001', 'test'),
    ('CQR_quantile',        'cqr_quantile',             'cqr'),
    ('CQR_quantile_ACI_g0.001', 'aci_cqr_quantile',     'cqr'),
    ('CQR_retrain',         'cqr_retrain',              'cqr'),
    ('CQR_retrain_ACI_g0.001',  'aci_cqr_retrain',      'cqr'),
    ('MoECP',               'moecp',                    'test'),
    ('MoECP_ACI_g0.001',    'aci_moecp',                'test'),
]

# These calibrators scale by the per-instance aleatoric variance, so run.py skips them
# without --prob_expert (run.py:582-609). On a MOE model they cannot exist at all, which
# is a different thing from "not run yet" -- they are written as n/a, never left blank.
NEEDS_PROB = {'cpvs', 'aleatoric_only', 'cp_aleatoric_scale',
              'aci_aleatoric_scale', 'aci_aleatoric_scale_g001',
              # MoECP consumes the gating distribution, refused without --prob_expert
              # (run.py:446-449).
              'moecp',
              # The ACI variants inherit their base's requirement exactly. aci_cp is
              # absent on purpose: like standard_cp it is ungated and runs on any model.
              # The two ACI CQR slugs are absent for the opposite reason -- they need
              # --use_quantile_loss, which is incompatible with --prob_expert, so they
              # live on the model_id=cqr column like their bases.
              'aci_cpvs', 'aci_aleatoric_only', 'aci_moecp'}

NA = 'n/a'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num-experts', default='3')
    ap.add_argument('--variant', default='MOG', choices=['MOG', 'MOE'])
    ap.add_argument('--seed', default='4021')
    ap.add_argument('--model', default='iTransformer')
    ap.add_argument('--out')
    args = ap.parse_args()

    tag = '' if args.variant == 'MOG' else '_moe'   # MOG is the historical default name
    out = args.out or os.path.join(
        ROOT, 'docs',
        f'table_seed{args.seed}_ne{args.num_experts}{tag}_{args.model.lower()}.csv')

    cells = {}
    with open(CSV) as fh:
        for r in csv.DictReader(fh):
            if (r['model'] == args.model and r['num_experts'] == args.num_experts
                    and r['seed'] == args.seed and r['features'] == 'M'
                    and r['unc_gating'] == '0' and r['coverage']):
                cells[(r['dataset'], int(r['pred_len']), r['method'],
                       r['variant'], r['model_id'])] = r

    header = ['dataset', 'pred_len', 'mse', 'mae']
    for label, *_ in METHODS:
        header += [f'{label}_coverage', f'{label}_width',
                   f'{label}_mse', f'{label}_mae']

    rows = []
    for d in DATASETS:
        pl = PL.get(d, 96)
        row = [d, pl]
        # mse/mae come from the model the MoG columns describe, i.e. the requested variant.
        err = cells.get((d, pl, 'standard_cp', args.variant, 'test'))
        row += [err['mse'], err['mae']] if err else ['', '']
        for _, slug, model_id in METHODS:
            if slug in NEEDS_PROB and args.variant == 'MOE':
                row += [NA, NA, NA, NA]
                continue
            variant = 'MOE' if model_id == 'cqr' else args.variant
            r = cells.get((d, pl, slug, variant, model_id))
            # Per-method mse/mae alongside the block-level pair above. The two differ for
            # the CQR columns, which live on the separate pinball-loss trunk: the block
            # value describes a model those rows are not computed from.
            row += ([r['coverage'], r['width'], r['mse'], r['mae']]
                    if r else ['', '', '', ''])
        rows.append(row)

    with open(out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)

    # Stride 4: each method contributes coverage/width/mse/mae, and coverage is the one
    # that decides whether the cell exists at all.
    applicable = sum(1 for row in rows for c in row[4::4] if c != NA)
    filled = sum(1 for row in rows for c in row[4::4] if c not in ('', NA))
    print(f'wrote {out} ({filled}/{applicable} applicable cells filled)')
    for row in rows:
        miss = [label for i, (label, *_) in enumerate(METHODS) if row[4 + 4 * i] == '']
        if miss:
            print(f'  MISSING {row[0]}: {", ".join(miss)}')


if __name__ == '__main__':
    main()
