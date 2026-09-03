"""Render docs/results_tables.tex from the result CSVs in docs/.

Inputs (all produced by other scripts -- this one only formats them):
  docs/table_seed4021_ne1_itransformer.csv      build_headline_table.py --num-experts 1
  docs/table_seed4021_ne1_moe_itransformer.csv  build_headline_table.py --num-experts 1 --variant MOE
  docs/table_seed4021_ne3_itransformer.csv      build_headline_table.py --num-experts 3
  docs/calibration_results_tsf.csv              collect_calibration_results.py (sweep only)

    python scripts/build_results_tex.py
    python scripts/build_results_tex.py --exclude weather --out docs/results_tables_no_weather.tex

--exclude drops dataset(s) from this run only (comma-separated), on top of the
standing EXCLUDE set below; --out picks where the .tex is written, so a variant
report can be built without touching the default docs/results_tables.tex. The
sweep tables (ETT only) are unaffected by --exclude unless an ETT dataset is named.

Layout: one row per (model, calibration) pair with the columns
Dataset / Model / Calibration / Coverage / Avg. Width. Every result table is
split into a base table (frozen calibrators, fixed alpha) and a separate ACI
table (ACI-adapted variants only, gamma=0.001 throughout, no base rows mixed
in) -- ETT, the remaining datasets, and the expert sweep, six tables in total.
Each table's caption states its full configuration (seed, alpha, gamma where
applicable, window size, L, H) so no table depends on reading another to be
interpreted.

The sweep is read from the collector output rather than from
docs/ett_expert_sweep_mog.csv because that file averages over seeds 4021-4025
and the table is wanted at seed 4021 only; the filters below are the ones
build_ett_expert_sweep.py applies, plus the seed.

By default every table is seed 4021 alone and no standard deviations are
reported. With --multiseed all six tables become mean +/- sample std over seeds
4021-4025 instead, written to docs/results_tables_multiseed.tex:

    python scripts/build_results_tex.py --multiseed \
        --out docs/results_tables_multiseed.tex

Seed coverage is uneven across methods, so a cell backed by fewer than five runs
is averaged over the runs it has, printed without a +/- term when only one
remains, and listed in a footnote under its table -- never silently padded.
"""
import argparse
import collections
import csv
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, 'docs')
OUT = os.path.join(DOCS, 'results_tables.tex')
MASTER = os.path.join(DOCS, 'calibration_results_tsf.csv')

NUM_WORDS = {5: 'Five', 6: 'Six', 7: 'Seven', 8: 'Eight'}

SEED = '4021'
TARGET = 0.90
COV_TOL = 0.01          # a row counts as "calibrated" if coverage >= TARGET - COV_TOL

DASH = '--'

ETT = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']

# Datasets collected but kept out of this report.
EXCLUDE = {'national-illness'}

# Calibration labels, keyed by the column prefix used in the headline CSVs. The
# ACI labels drop the "(gamma=...)" suffix on purpose: every ACI table now shows
# ACI rows only, at a single gamma stated once in the caption's config line, so
# repeating it on every row would be redundant. CP_MOG_ACI_g0.01 keeps its full
# label because it is dead code here (see MOG_COLS_ACI's comment) -- if it is
# ever reinstated it should not silently inherit the gamma=0.001 tables' short
# style.
CAL = {
    'CP': 'CP-fixed',
    'CP_ACI_g0.001': 'CP-fixed + ACI',
    'CPVS': 'CPVS',
    'CPVS_ACI_g0.001': 'CPVS + ACI',
    'CPVS_aleatoric': 'CPVS-aleatoric',
    'CPVS_aleatoric_ACI_g0.001': 'CPVS-aleatoric + ACI',
    'CP_MOG': 'CP-MoG',
    'CP_MOG_ACI_g0.01': r'CP-MoG + ACI ($\gamma$=0.01)',
    'CP_MOG_ACI_g0.001': 'CP-MoG + ACI',
    'MoECP': 'MoECP',
    'MoECP_ACI_g0.001': 'MoECP + ACI',
    'CQR_quantile': 'CQR',
    'CQR_quantile_ACI_g0.001': 'CQR + ACI',
    'CQR_retrain': 'CQR (retrained)',
    'CQR_retrain_ACI_g0.001': 'CQR (retrained) + ACI',
}

# Base (frozen-model) columns -- no ACI variant of anything appears here.
MOG_COLS_BASE = ['CP_MOG', 'CPVS', 'CPVS_aleatoric', 'MoECP', 'CP']

# Single Gaussian (K=1, prob_expert) drops MoECP/MoECP+ACI on purpose: MoECP's
# gate-localisation has nothing to localise over with one expert, so the row is
# excluded here rather than shown as a (degenerate) number -- MoGE keeps it.
SG_COLS_BASE = [c for c in MOG_COLS_BASE if c != 'MoECP']

# ACI-only columns, gamma=0.001 only: the base rows are dropped from this table
# entirely (they are already in the base table) and CP_MOG_ACI_g0.01 is left out
# -- every row here shares the single gamma stated in the table's config line,
# so mixing in the one method that historically also had a gamma=0.01 run would
# break that "one gamma per table" invariant.
MOG_COLS_ACI = ['CP_MOG_ACI_g0.001', 'CPVS_ACI_g0.001',
                'CPVS_aleatoric_ACI_g0.001', 'MoECP_ACI_g0.001',
                'CP_ACI_g0.001']
SG_COLS_ACI = [c for c in MOG_COLS_ACI if c != 'MoECP_ACI_g0.001']

SE_COLS_BASE = ['CP', 'CQR_quantile', 'CQR_retrain']
SE_COLS_ACI = ['CP_ACI_g0.001', 'CQR_quantile_ACI_g0.001',
               'CQR_retrain_ACI_g0.001']

# Calibrators whose row is dropped from a block when that block has no run for it,
# instead of being printed as a dash. Two different reasons produce this, kept in
# one set because the row-dropping logic is the same either way:
#  - base MoECP is now complete on MoGE and Single Gaussian for every dataset in
#    this report (2026-08-13 clip-to-window-max rerun), so this only still fires
#    on Single Expert, where the method is structurally undefined -- kept in the
#    set rather than removed so a future dataset gap degrades the same way;
#  - every ACI column here was only run on the small/medium datasets so far
#    (ACI-MoECP electricity/traffic is a permanent gap -- see FOOTNOTE_ACI;
#    ACI-CQR-retrain electricity/traffic is in flight -- see the dynamic note
#    appended to FOOTNOTE_ACI in main()).
_ACI_COLS = ['CP_ACI_g0.001', 'CPVS_ACI_g0.001', 'CPVS_aleatoric_ACI_g0.001',
             'MoECP_ACI_g0.001', 'CQR_quantile_ACI_g0.001', 'CQR_retrain_ACI_g0.001']
DROP_IF_MISSING = {'MoECP'} | {CAL[c] for c in _ACI_COLS}

# (model label, source CSV, calibration columns to take from it), in the order the
# rows appear inside each dataset block. The two CQR columns are the same
# pinball-loss model in all three CSVs, so they are listed once, under the model
# they actually describe.
BLOCKS_BASE = [
    ('MoGE', 'table_seed4021_ne3_itransformer.csv', MOG_COLS_BASE),
    ('Single Gaussian', 'table_seed4021_ne1_itransformer.csv', SG_COLS_BASE),
    ('Single Expert', 'table_seed4021_ne1_moe_itransformer.csv', SE_COLS_BASE),
]
BLOCKS_ACI = [
    ('MoGE', 'table_seed4021_ne3_itransformer.csv', MOG_COLS_ACI),
    ('Single Gaussian', 'table_seed4021_ne1_itransformer.csv', SG_COLS_ACI),
    ('Single Expert', 'table_seed4021_ne1_moe_itransformer.csv', SE_COLS_ACI),
]

# Multi-seed counterparts, from build_headline_table_multiseed.py. Same cells and same
# column lists as BLOCKS_BASE/BLOCKS_ACI -- only the source CSV differs, so either table
# can be switched between single-seed and mean +/- std without touching what it reports.
SEEDS_MULTI = ['4021', '4022', '4023', '4024', '4025']
BLOCKS_BASE_MS = [
    ('MoGE', 'table_multiseed_ne3_itransformer.csv', MOG_COLS_BASE),
    ('Single Gaussian', 'table_multiseed_ne1_itransformer.csv', SG_COLS_BASE),
    ('Single Expert', 'table_multiseed_ne1_moe_itransformer.csv', SE_COLS_BASE),
]
BLOCKS_ACI_MS = [
    ('MoGE', 'table_multiseed_ne3_itransformer.csv', MOG_COLS_ACI),
    ('Single Gaussian', 'table_multiseed_ne1_itransformer.csv', SG_COLS_ACI),
    ('Single Expert', 'table_multiseed_ne1_moe_itransformer.csv', SE_COLS_ACI),
]

# (dataset, channels, sampling interval, rows, split, horizon, batch size). Channel
# counts and batch sizes are the --enc_in / --batch_size of the run scripts; the
# splits are what data_provider/data_loader.py applies per loader class; the row
# counts and intervals are the csv files themselves.
DATASETS = [
    ('ETTh1', '7', 'hourly', '17420', '12/4/4 months', '96', '8'),
    ('ETTh2', '7', 'hourly', '17420', '12/4/4 months', '96', '8'),
    ('ETTm1', '7', '15 min', '69680', '12/4/4 months', '96', '8'),
    ('ETTm2', '7', '15 min', '69680', '12/4/4 months', '96', '8'),
    ('weather', '21', '10 min', '52696', '70/10/20', '96', '8'),
    ('electricity', '321', 'hourly', '26304', '70/10/20', '96', '4'),
    ('traffic', '862', 'hourly', '17544', '70/10/20', '96', '4'),
    ('exchange-rate', '8', 'daily', '7587', '70/10/20', '96', '8'),
]

# Sweep rows: collector method slug -> paper-facing name, in print order. The
# base names match build_ett_expert_sweep.py so the two outputs agree. MoECP now
# covers the full 20/20 (dataset, experts) grid (2026-08-13 clip-to-window-max
# rerun); dashes would still render like everywhere else in this report if a
# future gap reopened, since unlike the headline tables this table has no
# whole-model gap to distinguish it from.
# CQR/CQR (retrained) are a different model (model_id=cqr, no --prob_expert) but
# num_experts is still a real knob on that trunk, so they are included here too --
# see sweep_table()'s CSV filter for how their rows are admitted.
SWEEP_METHODS_BASE = [
    ('cp_aleatoric_scale', 'CP-MoG'),
    ('cpvs', 'CPVS'),
    ('aleatoric_only', 'CPVS-aleatoric'),
    ('standard_cp', 'CP-fixed'),
    ('moecp', 'MoECP'),
    ('cqr_quantile', 'CQR'),
    ('cqr_retrain', 'CQR (retrained)'),
]

# ACI-only, gamma=0.001 only: no base rows (already in the base sweep table),
# and aci_aleatoric_scale (gamma=0.01) is left out so every row shares the one
# gamma stated in the table's config line. aci_cp/aci_cpvs/aci_aleatoric_only have the
# full 1-5 expert sweep on ETT, matching aci_aleatoric_scale_g001 (the 2/4/5 gap
# was filled by scripts/run_aci_ett_ne245.sh). aci_moecp and the two ACI-CQR rows
# do not share that backfill -- they are ne=1,3 only, so most of their row is a
# dash; see the footnote below the ACI sweep table.
SWEEP_METHODS_ACI = [
    ('aci_aleatoric_scale_g001', 'CP-MoG + ACI'),
    ('aci_cpvs', 'CPVS + ACI'),
    ('aci_aleatoric_only', 'CPVS-aleatoric + ACI'),
    ('aci_cp', 'CP-fixed + ACI'),
    ('aci_moecp', 'MoECP + ACI'),
    ('aci_cqr_quantile', 'CQR + ACI'),
    ('aci_cqr_retrain', 'CQR (retrained) + ACI'),
]


def tex_escape(s):
    return s.replace('_', r'\_').replace('&', r'\&').replace('%', r'\%')


def num(x, nd):
    if x is None or x == '' or x == 'n/a':
        return None
    return round(float(x), nd)


def fmt(x, nd):
    v = num(x, nd)
    return DASH if v is None else f'{v:.{nd}f}'


def agg(vals):
    """(mean, sample std) over the runs that exist; std is None below two of them.

    Mirrors build_headline_table_multiseed.agg, which does the same job for the headline
    CSVs -- the sweep reads the collector output directly, so it has to aggregate here.
    """
    xs = [float(v) for v in vals if v not in (None, '', 'n/a')]
    if not xs:
        return None, None
    return statistics.fmean(xs), (statistics.stdev(xs) if len(xs) > 1 else None)


def load(fname):
    with open(os.path.join(DOCS, fname)) as fh:
        return {r['dataset']: r for r in csv.DictReader(fh)}


def emit_header(lines, spec, caption, label, head):
    lines.append(r'\begin{longtable}{%s}' % spec)
    lines.append('  \\caption{%s}' % caption)
    lines.append(r'  \label{%s} \\' % label)
    lines.append(r'  \toprule')
    lines.append('  ' + head)
    lines.append(r'  \midrule')
    lines.append(r'  \endfirsthead')
    lines.append(r'  \toprule')
    lines.append('  ' + head)
    lines.append(r'  \midrule')
    lines.append(r'  \endhead')
    lines.append(r'  \bottomrule')
    lines.append(r'  \endfoot')


def pm(mean, std, nd=4, bold=False):
    """'mean +/- std', or just the mean when a cell has a single seed (std is empty).

    A one-seed cell prints bare rather than as '+/- 0.0000': zero spread would claim the
    variance was measured and found to be nil, when in fact it was never measurable.

    Bolding goes through \\mathbf *inside* the math, not \\textbf around it: math mode
    selects its own font, so \\textbf{$x$} leaves the digits at normal weight and the
    highlight silently does nothing.
    """
    if mean in (None, '', 'n/a'):
        return DASH
    m = float(mean)
    if std in (None, '', 'n/a'):
        body = f'{m:.{nd}f}'
    else:
        body = r'%.*f \pm %.*f' % (nd, m, nd, float(std))
    return r'$\mathbf{%s}$' % body if bold else f'${body}$'


def headline_table_multiseed(datasets, caption, label, blocks):
    """Headline table aggregated over seeds: coverage/width/MSE/MAE as mean +/- std.

    MSE and MAE describe the backbone, not the calibrator, so they are constant down a
    model block and repeat on every row of it -- each row stays readable on its own.
    """
    data = {m: load(fname) for m, fname, _ in blocks}

    lines = []
    emit_header(
        lines, 'l l l r r r r', caption, label,
        r'\textbf{Dataset} & \textbf{Model} & \textbf{Calibration} & '
        r'\textbf{MSE} $\downarrow$ & \textbf{MAE} $\downarrow$ & '
        r'\textbf{Coverage} $\uparrow$ & \textbf{Avg. Width} $\downarrow$ \\')

    for di, ds in enumerate(datasets):
        block = []
        for model, _, cols in blocks:
            r = data[model].get(ds, {})
            for c in cols:
                # Per-method mse/mae, falling back to the block's backbone only when the
                # method has no error row of its own. The fallback is all-or-nothing per
                # (mean, std) pair: mixing a method's own mean with the block's std would
                # pair statistics from two different models -- and from two different seed
                # counts, since a 1-seed method has no std while the block may have five.
                if r.get(f'{c}_mse_mean') not in (None, '', 'n/a'):
                    mse_m, mse_s = r.get(f'{c}_mse_mean'), r.get(f'{c}_mse_std')
                else:
                    mse_m, mse_s = r.get('mse_mean'), r.get('mse_std')
                if r.get(f'{c}_mae_mean') not in (None, '', 'n/a'):
                    mae_m, mae_s = r.get(f'{c}_mae_mean'), r.get(f'{c}_mae_std')
                else:
                    mae_m, mae_s = r.get('mae_mean'), r.get('mae_std')
                block.append([
                    model, CAL[c], mse_m, mse_s, mae_m, mae_s,
                    num(r.get(f'{c}_coverage_mean'), 4), r.get(f'{c}_coverage_std'),
                    num(r.get(f'{c}_width_mean'), 4), r.get(f'{c}_width_std'),
                ])
        block = [b for b in block
                 if not (b[1] in DROP_IF_MISSING and b[6] is None and b[8] is None)]
        # Bold on the mean, under the same coverage floor as the single-seed tables; the
        # spread is reported but deliberately not part of the winner rule, so base and
        # multi-seed tables stay comparable in what "narrowest" means.
        ok = [i for i, b in enumerate(block)
              if b[6] is not None and b[8] is not None and b[6] >= TARGET - COV_TOL]
        best = min(ok, key=lambda i: block[i][8]) if ok else None

        if di:
            lines.append(r'  \midrule')
        for i, b in enumerate(block):
            model, cal = b[0], b[1]
            name = (r'\multirow{%d}{*}{%s}' % (len(block), tex_escape(ds))
                    if i == 0 else '')
            show_model = i == 0 or block[i - 1][0] != model
            w = pm(b[8], b[9], bold=(i == best))
            end = r' \\' if i == len(block) - 1 else r' \\*'
            lines.append('  %s & %s & %s & %s & %s & %s & %s%s'
                         % (name, model if show_model else '', cal,
                            pm(b[2], b[3]), pm(b[4], b[5]), pm(b[6], b[7]), w, end))

    lines.append(r'\end{longtable}')
    return '\n'.join(lines)


def headline_table(datasets, caption, label, blocks):
    data = {m: load(fname) for m, fname, _ in blocks}

    lines = []
    emit_header(
        lines, 'l l l r r r r', caption, label,
        r'\textbf{Dataset} & \textbf{Model} & \textbf{Calibration} & '
        r'\textbf{MSE} $\downarrow$ & \textbf{MAE} $\downarrow$ & '
        r'\textbf{Coverage} $\uparrow$ & \textbf{Avg. Width} $\downarrow$ \\')

    for di, ds in enumerate(datasets):
        # Collect the block first: the bold rule compares across models.
        block = []
        for model, _, cols in blocks:
            r = data[model].get(ds, {})
            for c in cols:
                # Per-method mse/mae where the method has its own run (the CQR columns
                # sit on a different model than their block), else the block's backbone.
                mse = r.get(f'{c}_mse') or r.get('mse')
                mae = r.get(f'{c}_mae') or r.get('mae')
                block.append([model, CAL[c],
                              num(r.get(f'{c}_coverage'), 4),
                              num(r.get(f'{c}_width'), 4),
                              mse, mae])
        block = [b for b in block
                 if not (b[1] in DROP_IF_MISSING and b[2] is None and b[3] is None)]
        ok = [i for i, b in enumerate(block)
              if b[2] is not None and b[3] is not None and b[2] >= TARGET - COV_TOL]
        best = min(ok, key=lambda i: block[i][3]) if ok else None

        if di:
            lines.append(r'  \midrule')
        for i, (model, cal, cov, wid, mse, mae) in enumerate(block):
            name = (r'\multirow{%d}{*}{%s}' % (len(block), tex_escape(ds))
                    if i == 0 else '')
            # A model label is written once, on the first row of its group.
            show_model = i == 0 or block[i - 1][0] != model
            w = DASH if wid is None else f'{wid:.4f}'
            if i == best:
                w = r'\textbf{%s}' % w
            end = r' \\' if i == len(block) - 1 else r' \\*'
            lines.append('  %s & %s & %s & %s & %s & %s & %s%s'
                         % (name, model if show_model else '', cal,
                            fmt(mse, 4), fmt(mae, 4),
                            DASH if cov is None else f'{cov:.4f}', w, end))

    lines.append(r'\end{longtable}')
    return '\n'.join(lines)


def dataset_table(datasets):
    word = NUM_WORDS.get(len(datasets), str(len(datasets))).lower()
    lines = [
        r'\begin{table}[t]',
        r'  \centering',
        r'  \caption{The %s benchmarks, as configured in the runs behind this '
        r'report. Channels are forecast jointly (\texttt{features=M}); rows is '
        r'the length of the raw series before splitting.}' % word,
        r'  \label{tab:data}',
        r'  \begin{tabular}{l r l r l r r}',
        r'    \toprule',
        r'    \textbf{Dataset} & \textbf{Channels} & \textbf{Interval} & '
        r'\textbf{Rows} & \textbf{Split} & \textbf{$H$} & \textbf{Batch} \\',
        r'    \midrule',
    ]
    for row in DATASETS:
        if row[0] not in datasets:
            continue
        lines.append('    ' + ' & '.join(tex_escape(c) for c in row) + r' \\')
    lines += [r'    \bottomrule', r'  \end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def sweep_table(methods, caption, label, seeds=(SEED,)):
    """One sweep table, one column pair per ETT dataset, H=96, MoG.

    CQR lives on a different slice of the CSV than everything else here (model_id=cqr,
    variant=MOE, since the pinball-loss trunk has no --prob_expert head) -- admitted
    alongside the MoG/model_id=test slice so its rows can share the same (dataset,
    method, experts) lookup as CP-MoG/CPVS/MoECP/etc.

    With more than one seed every cell becomes mean +/- sample std over the seeds that
    actually ran, matching the headline tables' aggregation exactly (agg/pm below); a
    cell backed by one seed prints its mean bare rather than claiming zero spread, and
    sweep_seed_note() lists those so a thin cell is never mistaken for a full one.
    """
    seeds = list(seeds)
    multiseed = len(seeds) > 1
    with open(MASTER) as fh:
        rows = [r for r in csv.DictReader(fh)
                if r['dataset'] in ETT
                and r['model'] == 'iTransformer'
                and ((r['variant'] == 'MOG' and r['model_id'] == 'test')
                     or (r['variant'] == 'MOE' and r['model_id'] == 'cqr'))
                and r['features'] == 'M' and r['unc_gating'] == '0'
                and r['pred_len'] == '96' and r['seed'] in seeds and r['coverage']]
    experts = sorted({r['num_experts'] for r in rows}, key=int)
    runs = collections.defaultdict(dict)
    for r in rows:
        # One run per (dataset, method, experts, seed); a later duplicate at the same
        # seed would be the same run recollected, so it overwrites rather than
        # double-counting into the mean.
        runs[(r['dataset'], r['method'], r['num_experts'])][r['seed']] = r

    # Four numeric columns per dataset (cov, w, MSE, MAE) puts this at 18 columns, which
    # does not fit the 2.2cm-margin portrait text block -- hence the landscape wrapper
    # below. The headline tables stay portrait; only the sweep needs the extra width.
    groups = ' '.join(r'& \multicolumn{4}{c}{\textbf{%s}}' % d for d in ETT)
    cmids = ' '.join(r'\cmidrule(lr){%d-%d}' % (3 + 4 * i, 6 + 4 * i)
                     for i in range(len(ETT)))
    head = '\n'.join([
        r'\multirow{2}{*}{\textbf{Experts}} & '
        r'\multirow{2}{*}{\textbf{Calibration}} %s \\' % groups,
        '  ' + cmids,
        r'  & %s \\' % ' '.join(
            [r'& cov $\uparrow$ & w $\downarrow$ & MSE $\downarrow$ & MAE $\downarrow$']
            * len(ETT))])

    lines = [r'\begin{landscape}', '{\\small']
    emit_header(
        lines, 'c l ' + ' '.join(['r r r r'] * len(ETT)), caption, label, head)

    for ei, e in enumerate(experts):
        block = []
        for slug, label_ in methods:
            # Each column of the row is one (mean, std) pair per metric; at a single
            # seed the mean is that run's own number and the std stays None.
            got = [list(runs.get((d, slug, e), {}).values()) for d in ETT]
            stats = []
            for g in got:
                # mse/mae come off the run rows directly, so the CQR rows carry their
                # own pinball trunk's error rather than the MoG backbone's.
                stats.append([agg([r[k] for r in g])
                              for k in ('coverage', 'width', 'mse', 'mae')])
            block.append([label_] + [[s[k] for s in stats] for k in range(4)])
        # The bold is per dataset column: widths of different datasets are not
        # comparable, so each column picks its own winner. The rule reads the mean
        # only, as in the headline tables, so bolding means the same thing in both.
        best = []
        for j in range(len(ETT)):
            ok = [i for i, b in enumerate(block)
                  if b[1][j][0] is not None and b[2][j][0] is not None
                  and b[1][j][0] >= TARGET - COV_TOL]
            best.append(min(ok, key=lambda i: block[i][2][j][0]) if ok else None)

        if ei:
            lines.append(r'  \midrule')
        for i, (label_, covs, wids, mses, maes) in enumerate(block):
            cells = [r'\multirow{%d}{*}{%s}' % (len(block), e) if i == 0 else '',
                     label_]
            for j in range(len(ETT)):
                if multiseed:
                    cells += [pm(covs[j][0], covs[j][1]),
                              pm(wids[j][0], wids[j][1], bold=(i == best[j])),
                              pm(mses[j][0], mses[j][1]), pm(maes[j][0], maes[j][1])]
                    continue
                w = fmt(wids[j][0], 4)
                if i == best[j] and w != DASH:
                    w = r'\textbf{%s}' % w
                cells += [fmt(covs[j][0], 4), w,
                          fmt(mses[j][0], 4), fmt(maes[j][0], 4)]
            end = r' \\' if i == len(block) - 1 else r' \\*'
            lines.append('  ' + ' & '.join(cells) + end)

    lines.append(r'\end{longtable}}')
    lines.append(r'\end{landscape}')
    return '\n'.join(lines)


PREAMBLE = r"""% docs/results_tables.tex -- generated by scripts/build_results_tex.py.
% Do not edit by hand: rerun
%     python scripts/build_results_tex.py
% Every number is copied verbatim from the CSVs listed in that script's
% docstring; the only editorial choices made here are the rounding and the
% bolding rule stated in each caption.
\documentclass[11pt]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{longtable}
\usepackage{pdflscape}
\usepackage{amsmath}
\setlength{\tabcolsep}{4pt}

\title{Calibration results: conformal methods on mixture-of-experts forecasters}
\author{}
\date{}

\begin{document}
\maketitle

\section{Experimental configuration}

\paragraph{Data.} Eight multivariate benchmarks (Table~\ref{tab:data}), $L=96$, standardised and never inverted, so widths are in standardised units.
"""

SETUP = r"""
\paragraph{Backbones.} Three iTransformer models: Single Expert ($K{=}1$, MSE), Single Gaussian ($K{=}1$, Gaussian NLL) and MoGE ($K{=}3$, softmax-gated).

\paragraph{Training.} Adam at learning rate $10^{-4}$ halved every epoch
(\texttt{lradj type1}), at most $10$ epochs with early stopping on validation
loss at patience $3$, batch size $8$ ($4$ for electricity and traffic, whose
channel counts make the larger batch infeasible). The mixture loss is the
gate-weighted sum of the per-expert losses -- Gaussian NLL for the
probabilistic models, MSE otherwise -- averaged over the batch, horizon and
channels. Every number in this report is a single training seed, $4021$.

The two CQR rows come from a \emph{different} model: the same trunk trained with
the pinball loss at quantiles $0.05$ and $0.95$ (\texttt{--use\_quantile\_loss},
\texttt{model\_id=cqr}), with no variance head. It is therefore the same model
whichever backbone column it is placed under.

\paragraph{Calibration protocol.} Every calibrator follows one online protocol.
The score window is initialised on the full validation split, and the test split
is then streamed one origin at a time. Feedback is \emph{delayed by the
horizon}: the pair $(\hat{y}_t, y_t)$ only enters the window at step $t+H$, and
the quantile is refreshed only once the window has rolled past the horizon, so
no interval is ever built from a target that would not yet have been observed at
that origin. The window holds the most recent $1000$ scores, and the nominal
level is $1-\alpha = 0.90$ throughout.
Calibration is per \emph{cell}: quantiles are taken along the sample axis, so
each (horizon step, channel) pair carries its own $q$.

\paragraph{Metrics.} Coverage is the fraction of (origin, horizon step, channel)
triples whose target falls inside its interval; average width is the mean of
$u - l$ over the same triples. The conformal quantile at window size $n$ is
always the $\lceil (n+1)(1-\alpha) \rceil / n$ empirical quantile with
\emph{higher} interpolation, the finite-sample-valid choice.

\section{Calibration methods}

All methods below produce a central interval at the same nominal $0.90$ level
and differ only in the non-conformity score and in where the quantile is read
off. With gate weights $\pi_k$ and per-expert moments $(\mu_k, \sigma_k^2)$ the
mixture predicts $\hat{y} = \sum_k \pi_k \mu_k$ and splits its variance into an
aleatoric $A = \sum_k \pi_k \sigma_k^2$ and an epistemic
$E = \sum_k \pi_k (\hat{y} - \mu_k)^2$; $E \equiv 0$ at $K=1$, which is why the
variance-scaled calibrators coincide in the Single Gaussian rows.

\paragraph{CP-fixed.} Split conformal on the point forecast: $s = |y -
\hat{y}|$, interval $\hat{y} \pm q$. It ignores the model's predictive variance
entirely, so its width is constant across all cells sharing a window -- the
baseline every other method has to beat on width without losing coverage.

\paragraph{CPVS.} Variance-scaled CP using the \emph{total} predictive standard
deviation $\sigma_{\text{tot}} = \sqrt{A + E}$:
\begin{equation*}
  s = \frac{|y - \hat{y}|}{\sigma_{\text{tot}} + \epsilon},
  \qquad \text{interval } \hat{y} \pm q\,\sigma_{\text{tot}}.
\end{equation*}
The interval now varies per cell with the model's own uncertainty, while the
conformal guarantee is preserved by calibrating the ratio.

\paragraph{CPVS-aleatoric.} The same idea driven by the aleatoric component
alone, on the squared scale: $s = (y - \hat{y})^2 / (A + \epsilon)$ and width
$\sqrt{q^2 A}$. The epistemic spread is deliberately discarded, which isolates
how much of CPVS's adaptivity comes from $A$.

\paragraph{CP-MoG.} Rescales only the aleatoric term and leaves the epistemic
term as an uncalibrated floor:
\begin{equation*}
  s = \max\!\left(0, \frac{(y-\hat{y})^2 - E}{A + \epsilon}\right),
  \qquad \text{width } = \sqrt{q^2 A + E}.
\end{equation*}
Clamping at zero costs nothing: since $q^2 \geq 0$ the event $\{s \leq q^2\}$ is
unchanged by the clamp, so the conformal guarantee survives it. This is the
method the mixture-density head exists for, and it is undefined without one.

\paragraph{MoECP.} Leaves the residual alone -- the score stays $|y - \hat{y}|$
-- and localises the \emph{quantile} instead. Calibration points whose gate
distribution resembles the test point's are up-weighted,
$w_i \propto \exp(-\tau\,\mathrm{KL}(\tilde{\pi} \,\|\, \pi(X_i)))$, and the
interval is the weighted quantile over those points. In the exact construction
the test point itself holds an atom of weight at $+\infty$, which is what makes
the construction finite-sample valid under weighted exchangeability; in these
tables, when the target level falls inside that atom the interval is instead
capped at the window's own largest weighted residual for that cell, the same
clip-to-window-max rule every other calibrator here already applies once
$1-\alpha$ exceeds what $n$ scores can express. This keeps width a single
number computed over the same population as every other row's -- the base
calibrators never produce an unbounded interval, so leaving MoECP's uncapped
would have made its coverage and width numbers not directly comparable to
theirs -- at the cost of exactness on the clipped cells, the same
accuracy-vs-comparability trade the +ACI recursion's $\alpha_t$ clip makes
below. The runs use $\tau = 1$: the tabular default $\tau = 100$ localises so
hard that the test point's own weight exceeds $\alpha$ often enough to make a
quarter of the intervals unbounded (pre-clip) on the hardest dataset, and
validity holds for every $\tau$, so lowering it trades sharpness of
localisation rather than coverage.

\paragraph{CQR and CQR (retrained).} Conformalised quantile regression on the
pinball-loss model: with predicted quantiles $[\ell, u]$ the score is
$s = \max(\ell - y,\; y - u)$ and the interval is $[\ell - q,\; u + q]$. The
retrained variant additionally fine-tunes the underlying quantile model as the
test stream advances -- every $H$ steps, for $3$ epochs, on a rolling window of
the last $1000$ observations -- so it is the only row here whose model weights
are not frozen at test time.

\paragraph{The +ACI variants.} Every method above pins its target level at the
nominal $\alpha$ and lets adaptivity come entirely from which scores sit in the
window. The +ACI variants instead make the working level a state variable,
updated online by the Adaptive Conformal Inference recursion of Gibbs \&
Cand\`es:
\begin{equation*}
  \alpha_{t+1} = \alpha_t + \gamma\,(\alpha - \mathrm{err}_t),
  \qquad \mathrm{err}_t = \mathbf{1}\{s_t > q_t\},
\end{equation*}
run independently per (horizon step, channel) so each cell tracks its own
miscoverage; the target $\alpha$ is never mutated, only $\alpha_t$. $\gamma = 0$
recovers the base method exactly. Every +ACI table in this report uses a single
step size, $\gamma = 0.001$ (the $0.1/H$ rule of thumb at $H = 96$), stated in
each table's caption rather than repeated per row; CP-MoG + ACI additionally
has a $\gamma = 0.01$ run from earlier work, not included in these tables
because it is the one case that visibly under-covers -- clipping $\alpha_t$ to
$[0,1]$ (next paragraph) removes the recursion's restoring force on cells the
window cannot cover, and $\gamma = 0.01$ moves fast enough to reach that regime
on some cells within the test stream, where $\gamma = 0.001$ mostly does not.
One deviation from the original recursion matters for every variant except
MoECP: $\alpha_t$ is clipped to $[0,1]$ so that intervals stay finite and width
remains a single comparable number. MoECP + ACI does not pay for this clip:
its localized quantile can itself return $+\infty$ when the target level
exceeds what the local window can express, which forces $\mathrm{err}_t = 0$
and pulls $\alpha_t$ back up, restoring the recursion's own correction
mechanism through the $+\infty$ branch instead of through an unbounded
$\alpha_t$.

\section{Headline calibration tables (seed 4021)}

Each dataset group below is reported as two tables: a base table with every
calibrator's own fixed-$\alpha$ number, and a separate ACI table with only the
ACI-adapted variants, at $\gamma = 0.001$ throughout -- the exact configuration
is stated in full in each table's caption, since the two tables are not meant
to require reading each other. \emph{CP-fixed} is split conformal on the point
forecast; \emph{CPVS} and \emph{CPVS-aleatoric} are its variance-scaled
variants; \emph{CP-MoG} scales the nonconformity score by the
mixture-of-Gaussians aleatoric variance; \emph{MoECP} calibrates per expert;
and \emph{CQR} is conformalised quantile regression with a pinball-loss model,
in a plain and a rolling-retrain form. MoGE is the three-expert
mixture-of-Gaussians head, Single Gaussian the one-expert one, and Single
Expert the plain MoE backbone.
"""

FOOTNOTE_BASE = (r'\noindent\footnotesize A dash is either a run that is missing or '
                 r'a method that cannot be defined for that model: the '
                 r'variance-scaled calibrators need the mixture-density head, so on '
                 r'Single Expert they do not exist at all. MoECP is omitted (not '
                 r'dashed) on both Single Expert, for the same reason, and Single '
                 r'Gaussian, by design: with one expert the gate distribution is '
                 r'the same for every point, so MoECP\textquotesingle s '
                 r'localisation has nothing to localise over and the row would '
                 r'only duplicate CP-fixed. MoGE has a MoECP row on every dataset '
                 r'in this report (2026-08-13 clip-to-window-max rerun; see the '
                 r'MoECP paragraph above). At one expert CPVS, CPVS-aleatoric and '
                 r'CP-MoG reduce to the same calibrator, hence the repeated numbers '
                 r'under Single Gaussian. The two CQR rows are a separate '
                 r'pinball-loss model and are identical whichever backbone they are '
                 r'compared against.\normalsize')

FOOTNOTE_ACI = (r'\noindent\footnotesize Dashes and omitted rows follow the base '
                r'table (previous footnote). MoECP + ACI has one further, permanent '
                r'gap the base MoECP row does not share: the base method supports '
                r'worker-parallel channels and reaches electricity and traffic, but '
                r'ACI-MoECP currently runs serially, which is impractical at '
                r'321/862 channels -- so MoECP + ACI is capped at 7/9 datasets '
                r'until worker support is added, not merely pending a '
                r'run.\normalsize')


MIDDLE = r"""
\section{Expert sweep on ETT (MoG variant)}

Tables~\ref{tab:sweep_base} and \ref{tab:sweep_aci} track the same calibrators,
one column pair per ETT dataset, as the number of mixture components grows from
one to five. Coverage is flat across mixture sizes for every calibrator, so the
widths are directly comparable down a column. CQR and CQR (retrained) sit on
the separate pinball-loss trunk (as in the headline tables above), not the
MoG model the other rows read off, but \texttt{num\_experts} is still a real
setting on that trunk, so they are swept the same way. A dash marks a cell
with no run, per the footnotes below each table.
"""

FOOTNOTE_SWEEP_BASE = (r'\noindent\footnotesize A dash marks a cell with no '
                        r'run. MoECP now covers the full $1$--$5$ expert grid '
                        r'on all four ETT datasets (2026-08-13 '
                        r'clip-to-window-max rerun; see the MoECP paragraph '
                        r'above); CQR and CQR (retrained) are likewise '
                        r'complete across all five expert counts.\normalsize')

FOOTNOTE_SWEEP_ACI = (r'\noindent\footnotesize Dashes follow the base table '
                       r'(previous footnote), plus one more gap specific to '
                       r'this table: MoECP + ACI was only run at $1$ and $3$ '
                       r'experts (the project\textquotesingle s headline grids), '
                       r'not the full $1$--$5$ sweep the other ACI rows have -- '
                       r'so that row is dashed at $2$, $4$ and $5$ experts, '
                       r'marking the gap rather than hiding it.\normalsize')

# Full run configuration, spelled out in every caption rather than left to the
# prose sections, so each table is self-contained. The ACI config adds gamma;
# nothing else differs between the two tables of a pair.
CONFIG_BASE = r'iTransformer, multivariate, seed $4021$, $\alpha=0.1$, window $=1000$, $L=96$'
CONFIG_ACI = r'iTransformer, multivariate, seed $4021$, $\alpha=0.1$, $\gamma=0.001$, window $=1000$, $L=96$'


def caption_base(scope, h_str):
    return (r'Efficiency of the conformal prediction intervals at a $0.90$ '
            r'target coverage, %s, base calibrators (fixed $\alpha$). '
            r'Configuration: %s, %s. Bold marks the narrowest interval among '
            r'the rows of a dataset that still reach coverage $\geq %.2f$.'
            % (scope, CONFIG_BASE, h_str, TARGET - COV_TOL))


def caption_aci(scope, h_str, base_label):
    return (r'ACI-adapted calibration intervals at a $0.90$ target coverage, '
            r'%s -- the base (fixed-$\alpha$) numbers for the same cells are '
            r'in Table~\ref{%s}. Configuration: %s, %s. Bold is recomputed '
            r'over this table\textquotesingle s own rows, so the winner can '
            r'differ from the base table. $\gamma=0.01$ is not shown here; '
            r'see the "+ACI variants" paragraph above.'
            % (scope, base_label, CONFIG_ACI, h_str))


CONFIG_BASE_MS = (r'iTransformer, multivariate, seeds $4021$--$4025$, $\alpha=0.1$, '
                  r'window $=1000$, $L=96$')
CONFIG_ACI_MS = (r'iTransformer, multivariate, seeds $4021$--$4025$, $\alpha=0.1$, '
                 r'$\gamma=0.001$, window $=1000$, $L=96$')

# Appended to the multi-seed captions: says once what the +/- is and what bolding does,
# so neither table depends on the surrounding prose to be read correctly.
MS_CAPTION_TAIL = (r'MSE and MAE describe the backbone rather than the calibrator, so '
                   r'they are constant within a model block and repeat on each of its '
                   r'rows. Bold marks the narrowest mean width among the rows of a '
                   r'dataset whose mean coverage still reaches $\geq %.2f$; the spread '
                   r'is reported but does not enter that rule. A cell printed without a '
                   r'$\pm$ term has only one seed so far (see the footnote).'
                   % (TARGET - COV_TOL))


def partial_seed_note(blocks, datasets, n_seeds):
    """Footnote listing cells backed by fewer than n_seeds runs, computed from the CSVs.

    Printed rather than silently averaged: a mean over 2 seeds and a mean over 5 are not
    the same claim, and the +/- term alone does not distinguish them. Returns '' once every
    reported cell is complete, so the note disappears on its own.
    """
    short = collections.defaultdict(set)
    for model, fname, cols in blocks:
        try:
            data = load(fname)
        except FileNotFoundError:
            continue
        for ds in datasets:
            r = data.get(ds, {})
            for c in cols:
                n = r.get(f'{c}_n_seeds')
                if n in (None, '', 'n/a'):
                    continue
                if 0 < int(n) < n_seeds:
                    short[f'{CAL[c]} ({int(n)}/{n_seeds})'].add(ds)
    if not short:
        return ''
    bits = ', '.join(f'{k} on {", ".join(sorted(v))}'
                     for k, v in sorted(short.items()))
    return (r'\par\noindent\footnotesize Not every cell has all %d seeds yet; these are '
            r'averaged over fewer runs and print without a $\pm$ term when only one '
            r'remains: %s.\normalsize' % (n_seeds, tex_escape(bits)))


def sweep_seed_note(methods, seeds):
    """partial_seed_note's counterpart for the sweep, counted off the collector CSV.

    The sweep does not go through the multiseed headline CSVs, so it has no `_n_seeds`
    column to read; the counts are recomputed here from the same slice sweep_table()
    selects. Rows short on every expert count are reported once as "all experts" rather
    than five times, since that is the common case (a method run only at seed 4021).
    """
    seeds = set(seeds)
    with open(MASTER) as fh:
        rows = [r for r in csv.DictReader(fh)
                if r['dataset'] in ETT and r['model'] == 'iTransformer'
                and ((r['variant'] == 'MOG' and r['model_id'] == 'test')
                     or (r['variant'] == 'MOE' and r['model_id'] == 'cqr'))
                and r['features'] == 'M' and r['unc_gating'] == '0'
                and r['pred_len'] == '96' and r['seed'] in seeds and r['coverage']]
    experts = sorted({r['num_experts'] for r in rows}, key=int)
    seen = collections.defaultdict(set)
    for r in rows:
        seen[(r['dataset'], r['method'], r['num_experts'])].add(r['seed'])

    bits = []
    for slug, label_ in methods:
        short = sorted({e for e in experts for d in ETT
                        if 0 < len(seen[(d, slug, e)]) < len(seeds)}, key=int)
        if not short:
            continue
        where = ('all expert counts' if len(short) == len(experts)
                 else ('expert ' if len(short) == 1 else 'experts ') + ', '.join(short))
        bits.append(f'{label_} on {where}')
    if not bits:
        return ''
    return (r'\par\noindent\footnotesize Not every cell here has all %d seeds; those '
            r'are averaged over fewer runs and print without a $\pm$ term when only '
            r'one remains: %s.\normalsize' % (len(seeds), tex_escape('; '.join(bits))))


def caption_base_ms(scope, h_str, n_seeds):
    return (r'Efficiency of the conformal prediction intervals at a $0.90$ target '
            r'coverage, %s, base calibrators (fixed $\alpha$), aggregated over %d '
            r'training seeds as mean $\pm$ sample standard deviation. '
            r'Configuration: %s, %s. %s'
            % (scope, n_seeds, CONFIG_BASE_MS, h_str, MS_CAPTION_TAIL))


def caption_aci_ms(scope, h_str, base_label, n_seeds):
    return (r'ACI-adapted calibration intervals at a $0.90$ target coverage, '
            r'%s, aggregated over %d training seeds as mean $\pm$ sample standard '
            r'deviation -- the base (fixed-$\alpha$) numbers for the '
            r'same cells are in Table~\ref{%s}. Configuration: %s, %s. %s '
            r'$\gamma=0.01$ is not shown here; see the '
            r'"+ACI variants" paragraph above.'
            % (scope, n_seeds, base_label, CONFIG_ACI_MS, h_str, MS_CAPTION_TAIL))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exclude', default='',
                     help='comma-separated dataset names to drop from this '
                          'run only, on top of the standing EXCLUDE set '
                          '(e.g. --exclude weather)')
    ap.add_argument('--out', default=OUT,
                     help='output path, for building a variant report '
                          'alongside the default docs/results_tables.tex')
    ap.add_argument('--multiseed', action='store_true',
                     help='render every result table (base, ACI and both expert '
                          'sweeps) as mean +/- std over seeds 4021-4025 instead of '
                          'seed 4021 alone. The headline tables read the '
                          'table_multiseed_*.csv files from '
                          'build_headline_table_multiseed.py; the sweeps aggregate '
                          'the collector CSV directly. Cells with fewer seeds print '
                          'their mean bare and are listed in a footnote.')
    args = ap.parse_args()
    exclude = EXCLUDE | {d.strip() for d in args.exclude.split(',') if d.strip()}

    ett = [d for d in ETT if d not in exclude]
    other = [d for d in load(BLOCKS_BASE[0][1])
             if d not in ETT and d not in exclude]
    all_datasets = ett + other
    word = NUM_WORDS.get(len(all_datasets), str(len(all_datasets)))
    preamble = PREAMBLE.replace('Eight multivariate benchmarks',
                                 f'{word} multivariate benchmarks')

    # If ACI-CQR-retrain has landed on electricity/traffic by the time this runs,
    # this note quietly stops applying next regen -- it is computed, not hardcoded.
    se = load(BLOCKS_ACI[2][1])
    retrain_gaps = [d for d in ('electricity', 'traffic')
                    if d in se and not se[d].get('CQR_retrain_ACI_g0.001_coverage')]
    footnote_aci = FOOTNOTE_ACI
    if retrain_gaps:
        footnote_aci += (
            r'\par\noindent\footnotesize CQR (retrained) + ACI is additionally '
            r'still in flight on %s at the time of writing; those rows are '
            r'omitted rather than dashed until the runs land.\normalsize'
            % ' and '.join(retrain_gaps))

    sweep_scope = ('one column pair per ETT dataset (cov: empirical coverage; '
                   'w: average interval width)')

    if args.multiseed:
        n = len(SEEDS_MULTI)
        setup = SETUP.replace(
            'Every number in this report is a single training seed, $4021$.',
            'Every number in this report is the mean over five training seeds, '
            '$4021$--$4025$, plus or minus the sample standard deviation across '
            'them; a cell whose runs do not all exist yet is averaged over the '
            'seeds that do and says so in that table\'s footnote.').replace(
            r'\section{Headline calibration tables (seed 4021)}',
            r'\section{Headline calibration tables (seeds 4021--4025)}')

        footnote_base_tab = FOOTNOTE_BASE + partial_seed_note(
            BLOCKS_BASE_MS, all_datasets, n)
        footnote_aci_tab = footnote_aci + partial_seed_note(
            BLOCKS_ACI_MS, all_datasets, n)

        base_ett = headline_table_multiseed(
            ett, caption_base_ms('ETT datasets', '$H=96$', n),
            'tab:ett_base', BLOCKS_BASE_MS)
        base_other = headline_table_multiseed(
            other, caption_base_ms('the remaining datasets', '$H=96$', n),
            'tab:other_base', BLOCKS_BASE_MS)
        aci_ett = headline_table_multiseed(
            ett, caption_aci_ms('ETT datasets', '$H=96$', 'tab:ett_base', n),
            'tab:ett_aci', BLOCKS_ACI_MS)
        aci_other = headline_table_multiseed(
            other, caption_aci_ms('the remaining datasets', '$H=96$',
                                  'tab:other_base', n),
            'tab:other_aci', BLOCKS_ACI_MS)

        sweep_base = sweep_table(
            SWEEP_METHODS_BASE, caption_base_ms(sweep_scope, '$H=96$', n),
            'tab:sweep_base', SEEDS_MULTI)
        sweep_aci = sweep_table(
            SWEEP_METHODS_ACI,
            caption_aci_ms(sweep_scope, '$H=96$', 'tab:sweep_base', n),
            'tab:sweep_aci', SEEDS_MULTI)
        footnote_sweep_base = FOOTNOTE_SWEEP_BASE + sweep_seed_note(
            SWEEP_METHODS_BASE, SEEDS_MULTI)
        footnote_sweep_aci = FOOTNOTE_SWEEP_ACI + sweep_seed_note(
            SWEEP_METHODS_ACI, SEEDS_MULTI)
    else:
        setup = SETUP
        footnote_base_tab = FOOTNOTE_BASE
        footnote_aci_tab = footnote_aci

        base_ett = headline_table(
            ett, caption_base('ETT datasets', '$H=96$'),
            'tab:ett_base', BLOCKS_BASE)
        base_other = headline_table(
            other, caption_base('the remaining datasets', '$H=96$'),
            'tab:other_base', BLOCKS_BASE)
        aci_ett = headline_table(
            ett, caption_aci('ETT datasets', '$H=96$', 'tab:ett_base'),
            'tab:ett_aci', BLOCKS_ACI)
        aci_other = headline_table(
            other, caption_aci('the remaining datasets', '$H=96$',
                               'tab:other_base'),
            'tab:other_aci', BLOCKS_ACI)

        sweep_base = sweep_table(
            SWEEP_METHODS_BASE,
            caption_base(sweep_scope + ', one run per cell', '$H=96$'),
            'tab:sweep_base')
        sweep_aci = sweep_table(
            SWEEP_METHODS_ACI,
            caption_aci(sweep_scope + ', one run per cell', '$H=96$',
                        'tab:sweep_base'),
            'tab:sweep_aci')
        footnote_sweep_base = FOOTNOTE_SWEEP_BASE
        footnote_sweep_aci = FOOTNOTE_SWEEP_ACI

    parts = [
        preamble, dataset_table(all_datasets), setup,
        base_ett,
        '', footnote_base_tab, '',
        aci_ett,
        '', footnote_aci_tab, '',
        base_other,
        '', footnote_base_tab, '',
        aci_other,
        '', footnote_aci_tab, '',
        MIDDLE,
        sweep_base,
        '', footnote_sweep_base, '',
        sweep_aci,
        '', footnote_sweep_aci, '',
        r'\end{document}',
    ]
    with open(args.out, 'w') as fh:
        fh.write('\n'.join(parts) + '\n')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
