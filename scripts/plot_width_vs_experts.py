"""Plot mean prediction-interval width vs. number of experts, with seed error bars.

Reads the raw calibration records (`result_calibration_*.txt` written by the
calibration code, plus `logs/**/*.log` for runs that only reached stdout), keeps the
rows matching one (dataset, backbone, pred_len, variant, window) cell, and averages
'Avg Width' over the target seeds for each (method, num_experts).

Default figure -- the one asked for -- is ETTh1 / iTransformer / pred_len 96,
experts 1-5, seeds 4021-4025, sliding window 1000:

    python scripts/plot_width_vs_experts.py

Writes figures/width_vs_experts_<dataset>_<model>_pl<pred_len>.{png,csv}. The CSV is
the table view that the low-contrast palette slots require, and it carries the
per-seed values so a point can be traced back to its run. Re-run after new
experiments land -- it always re-parses the sources, never a cached summary.
"""
import argparse
import collections
import glob
import os
import re
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Same setting grammar as run.py:281 / scripts/collect_calibration_results.py.
SETTING_RE = re.compile(
    r'^long_term_forecast_(?P<model_id>.+?)_(?P<model>[A-Za-z0-9]+)_(?P<dataset>.+?)'
    r'_ne(?P<ne>\d+)_pe(?P<pe>\d+)_ug(?P<ug>\d+)_ft(?P<ft>[A-Z]+)_sl(?P<sl>\d+)_ll(?P<ll>\d+)'
    r'_pl(?P<pl>\d+)_dm(?P<dm>\d+)_nh(?P<nh>\d+)_el(?P<el>\d+)_dl(?P<dl>\d+)_df(?P<df>\d+)'
    r'_expand(?P<expand>\d+)_dc(?P<dc>\d+)_fc(?P<fc>\d+)_eb(?P<eb>\w+?)_dt(?P<dt>True|False)'
    r'_(?P<des>.+)_(?P<itr>\d+)_seed(?P<seed>\d+)$')

NUM = r'-?\d+\.?\d*(?:e-?\d+)?|nan'
SET_RE = re.compile(r'(long_term_forecast_\S*?)(?:<<<|\.\.\.|\s|$)')
RES_RE = re.compile(r'^([A-Za-z][A-Za-z0-9_() .,\'^*=+-]*?) Results(?: \(([^)]*)\))?:\s*$')
WINDOW_RE = re.compile(r'window[=:]\s*(\d+)')

# ------------------------------------------------------------------ the series
# One entry per line on the chart. `match` is tested against the lowercased method
# label as printed by the calibration code; `variant` is (prob_expert, unc_gating)
# and `model_id` pins the training run the calibration was applied to (the CQR
# variants were trained under --model_id cqr with a quantile head, everything else
# under the shared `test` runs).
#
# Colors are categorical slots 1-5 of the design-system palette, assigned in the
# vertical order the lines come out in so that every visually adjacent pair is a
# validated adjacent pair. Marker + dash style repeat the identity, which is what
# lets the three sub-3:1 slots (aqua/yellow/magenta) ship on a white surface.
def _base(*needles):
    """Match a base method's label, excluding its ACI variant.

    Every ACI label contains its base method's keyword ("ACI MoECP" contains "moecp",
    "ACI Retrained CQR" contains "retrained cqr", ...), so a bare substring test silently
    folds ACI runs into the base method's line -- two methods averaged into one series,
    with no error. The token test matches scripts/collect_calibration_results.py's
    norm_method, which guards the same collision.
    """
    return lambda l: any(n in l for n in needles) and 'aci' not in l.split()


SERIES = [
    dict(key='cqr_quantile', display='CQR (standard)',
         match=_base('cqr quantile'),
         model_id='cqr', variant=('0', '0'),
         color='#2a78d6', marker='o', ls='-'),
    dict(key='cqr_retrain', display='CQR (rolling retrain)',
         match=_base('retrained cqr'),
         model_id='cqr', variant=('0', '0'),
         color='#eb6834', marker='s', ls='--'),
    # Plain CP is the only method here that is not tied to a mixture-density head, so
    # it is the one whose backbone is a choice: --cp_variant overwrites this entry.
    dict(key='standard_cp', display='CP',
         match=_base('standard cp'),
         model_id='test', variant=('0', '0'),
         color='#1baf7a', marker='^', ls='-.'),
    dict(key='moecp', display='MoECP',
         match=_base('moecp'),
         model_id='test', variant=('1', '0'),
         color='#eda100', marker='D', ls=(0, (3, 1, 1, 1, 1, 1))),
    dict(key='aleatoric_scale', display='CP-MOG',
         match=_base('aleatoric scale'),
         model_id='test', variant=('1', '0'),
         color='#e87ba4', marker='v', ls=':'),
]

# The ACI variants are deliberately not plotted here. The palette above is five
# categorical slots chosen so that every visually adjacent pair is a validated adjacent
# pair; adding six more lines would break that and double the ink on a figure whose point
# is the expert-count trend, not the calibrator comparison. The ACI numbers live in
# docs/calibration_results_tsf.md.

VARIANT_NAME = {('0', '0'): 'MoE', ('1', '0'): 'MoG', ('1', '1'): 'MoGU'}
VARIANT_CODE = {v: k for k, v in VARIANT_NAME.items()}


def spec_of(key):
    return next(s for s in SERIES if s['key'] == key)


def labelled(spec):
    """Display name with the backbone appended -- unless the name already carries it
    (CP-MOG would otherwise read 'CP-MOG · MoG')."""
    variant = VARIANT_NAME.get(spec['variant'], '?')
    return spec['display'] if spec['display'].lower().endswith(variant.lower()) \
        else f'{spec["display"]} · {variant}'

# Every calibrator plotted here is constructed with window_size=1000 in
# exp/exp_long_term_forecasting.py (retrained CQR takes it from --retrain_window,
# default 1000, and prints it). A record whose label or run header states a
# different window is dropped rather than silently averaged in.
DEFAULT_WINDOW = 1000


def classify(label):
    """Map a printed method label onto one of SERIES, or None to ignore the record."""
    l = label.lower()
    if 'diagnostics' in l or 'adaptive window' in l:
        return None
    for s in SERIES:
        if s['match'](l):
            return s['key']
    return None


def collect(args):
    """(method_key, num_experts, seed) -> {'width': float, 'source': str}.

    Both sources are scanned; a value from a result_*.txt file wins over a log,
    and among equals the last one read wins (re-runs supersede earlier ones), which
    is the same precedence scripts/collect_calibration_results.py applies.
    """
    seeds = {str(s) for s in args.seeds}
    experts = {str(e) for e in args.experts}
    obs = collections.defaultdict(list)   # key -> [(is_txt, width, source, window)]
    skipped_window = collections.Counter()

    def keep(cfg, key, window):
        spec = next(s for s in SERIES if s['key'] == key)
        if cfg['dataset'] != args.dataset or cfg['model'] != args.model:
            return False
        if cfg['pl'] != str(args.pred_len) or cfg['seed'] not in seeds or cfg['ne'] not in experts:
            return False
        if cfg['model_id'] != spec['model_id'] or (cfg['pe'], cfg['ug']) != spec['variant']:
            return False
        if window is not None and window != args.window:
            skipped_window[(key, window)] += 1
            return False
        return True

    # -- result_calibration_*.txt: "<setting> (<label>)" then a metrics line
    for path in sorted(glob.glob(os.path.join(ROOT, 'result_calibration_*.txt'))):
        src = os.path.basename(path)
        lines = open(path, errors='ignore').read().splitlines()
        for i, line in enumerate(lines):
            m = re.match(r'^(long_term_forecast_\S+)\s+\((.*)\)$', line.strip())
            if not m:
                continue
            cfg = SETTING_RE.match(m.group(1))
            key = classify(m.group(2))
            if not cfg or key is None:
                continue
            w = WINDOW_RE.search(m.group(2))
            window = int(w.group(1)) if w else None
            if not keep(cfg.groupdict(), key, window):
                continue
            body = lines[i + 1].strip() if i + 1 < len(lines) else ''
            width = re.search(r'(?<!Median )Width:\s*(' + NUM + ')', body)
            if width:
                obs[(key, int(cfg.group('ne')), int(cfg.group('seed')))].append(
                    (True, float(width.group(1)), src, window))

    # -- logs/**/*.log: "<Method> Results:" block under the last announced setting
    log_files = sorted(glob.glob(os.path.join(ROOT, 'logs', '*.log')) +
                       glob.glob(os.path.join(ROOT, 'logs', '*', '*.log')))
    for path in log_files:
        src = 'logs/' + os.path.relpath(path, os.path.join(ROOT, 'logs'))
        cur, window = None, None
        lines = open(path, errors='ignore').read().splitlines()
        for i, line in enumerate(lines):
            s = line.strip()
            hit = SET_RE.search(s)
            if hit and ('calibrating' in s or 'testing' in s or ' for ' in s or 'training' in s):
                cand = SETTING_RE.match(hit.group(1).rstrip('.'))
                if cand:
                    cur, window = cand, None
            if s.startswith('Starting ') or s.startswith('>>>>>>> Start '):
                w = WINDOW_RE.search(s)
                window = int(w.group(1)) if w else window
            m = RES_RE.match(s)
            if not m or cur is None:
                continue
            label = m.group(1) + (' (' + m.group(2) + ')' if m.group(2) else '')
            key = classify(label)
            if key is None:
                continue
            w = WINDOW_RE.search(label)
            run_window = int(w.group(1)) if w else window
            if not keep(cur.groupdict(), key, run_window):
                continue
            block = '\n'.join(lines[i + 1:i + 8]).split('\n\n')[0]
            width = re.search(r'Avg Width:\s*(' + NUM + ')', block)
            if width:
                obs[(key, int(cur.group('ne')), int(cur.group('seed')))].append(
                    (False, float(width.group(1)), src, run_window))

    picked = {}
    for k, v in obs.items():
        txt = [o for o in v if o[0]]
        is_txt, width, src, window = (txt or v)[-1]
        spread = max(o[1] for o in v) - min(o[1] for o in v)
        picked[k] = dict(width=width, source=src, n_obs=len(v), spread=spread)
    if skipped_window:
        for (key, window), n in sorted(skipped_window.items()):
            print(f'  note: dropped {n} {key} record(s) with window={window} '
                  f'(keeping window={args.window} only)')
    return picked


def aggregate(picked, args):
    """(method_key, num_experts) -> mean / std / n over the target seeds."""
    stats = {}
    for spec in SERIES:
        for ne in args.experts:
            vals = [(s, picked[(spec['key'], ne, s)]['width'])
                    for s in args.seeds if (spec['key'], ne, s) in picked]
            if not vals:
                continue
            w = [v for _, v in vals]
            stats[(spec['key'], ne)] = dict(
                mean=statistics.fmean(w),
                std=statistics.stdev(w) if len(w) > 1 else 0.0,
                n=len(w),
                per_seed=dict(vals))
    return stats


def tight_cluster(stats, args):
    """The series whose seed s.d. the shared axis swallows -- candidates for the detail panel.

    CP-family widths sit within ~0.2 of each other while CQR runs ~1.5 above them, so on
    one axis a ±0.007 seed s.d. is a sub-pixel bar. Returns [] when the shared axis
    already shows every error bar, in which case the detail panel is not drawn.
    """
    means = [st['mean'] for st in stats.values()]
    span = max(means) - min(means)
    if not span:
        return []
    tight = [s for s in SERIES
             if any((s['key'], ne) in stats for ne in args.experts)
             and max(stats[(s['key'], ne)]['std'] for ne in args.experts
                     if (s['key'], ne) in stats) < 0.015 * span]
    if len(tight) < 2:
        return []
    pts = [stats[(s['key'], ne)] for s in tight for ne in args.experts
           if (s['key'], ne) in stats]
    lo = min(p['mean'] - p['std'] for p in pts)
    hi = max(p['mean'] + p['std'] for p in pts)
    return tight if hi - lo <= 0.35 * span else []


def draw(ax, series, stats, args, full, label=True):
    for spec in series:
        xs = [ne for ne in args.experts if (spec['key'], ne) in stats]
        if not xs:
            if label:
                print(f"  warning: no data for '{spec['display']}' -- omitted from the figure")
            continue
        seen = sum(stats[(spec['key'], ne)]['n'] for ne in xs)
        # The backbone rides in the label: which one a method was calibrated on is a
        # real confound between the lines, not a footnote. And a partially-filled
        # series would otherwise read as a finished measurement, so say how many of
        # its (experts, seed) runs actually exist.
        name = labelled(spec)
        if seen != full:
            name += f' ({seen}/{full} runs)'
        # A single-seed cell has no spread to show; a zero-length bar would claim it
        # was reproduced exactly. NaN draws no bar at all.
        ax.errorbar(xs, [stats[(spec['key'], ne)]['mean'] for ne in xs],
                    yerr=[stats[(spec['key'], ne)]['std'] if stats[(spec['key'], ne)]['n'] > 1
                          else float('nan') for ne in xs],
                    capsize=4, capthick=1.2, elinewidth=1.2,
                    color=spec['color'], marker=spec['marker'], markersize=7,
                    markeredgecolor='#fcfcfb', markeredgewidth=1.0,
                    linestyle=spec['ls'], linewidth=2.0,
                    label=name if label else None, zorder=3)

    ax.set_xticks(list(args.experts))
    ax.set_xlim(min(args.experts) - 0.25, max(args.experts) + 0.25)
    ax.set_facecolor('#fcfcfb')
    ax.grid(True, axis='y', color='#e3e2de', linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#c8c7c2')
    ax.tick_params(colors='#52514e', labelsize=9)


def plot(stats, args, out_png):
    full = len(args.seeds) * len(args.experts)
    detail = tight_cluster(stats, args)
    # The detail panel is its own axes rather than an inset: an inset large enough to
    # separate the CP-family lines lands on top of the CQR error bars.
    fig, axes = plt.subplots(1, 2 if detail else 1, figsize=(9.4 if detail else 6.4, 4.6),
                             gridspec_kw={'width_ratios': [1.4, 1]} if detail else None)
    fig.patch.set_facecolor('#fcfcfb')
    ax = axes[0] if detail else axes

    draw(ax, SERIES, stats, args, full)
    ax.set_xlabel('Number of experts', fontsize=11, color='#0b0b0b')
    ax.set_ylabel('Mean average interval width', fontsize=11, color='#0b0b0b')
    ax.set_title('all methods', fontsize=10, color='#52514e', pad=8)

    if detail:
        draw(axes[1], detail, stats, args, full, label=False)
        axes[1].set_xlabel('Number of experts', fontsize=11, color='#0b0b0b')
        axes[1].set_title('detail — ' + ', '.join(s['display'] for s in detail),
                          fontsize=10, color='#52514e', pad=8)

    fig.suptitle(f'Prediction-interval width vs. number of experts   ·   '
                 f'{args.dataset} · {args.model} · horizon {args.pred_len}',
                 fontsize=12.5, color='#0b0b0b', y=0.98)

    # Legend under the axes: with five series any in-plot placement collides with the
    # CQR band at the top or the three CP-family lines at the bottom.
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, ncol=3, labelcolor='#52514e',
               loc='lower center', bbox_to_anchor=(0.5, 0.085), columnspacing=1.6,
               handlelength=2.6)
    fig.text(0.5, 0.048, f'mean ± 1 s.d. over seeds {min(args.seeds)}–{max(args.seeds)} '
                         f'(n={len(args.seeds)}); calibration sliding window {args.window}',
             ha='center', fontsize=7.5, color='#52514e')
    fig.text(0.5, 0.012, 'each label names the backbone the method was calibrated on; '
                         'CQR is read off its own quantile-head training runs',
             ha='center', fontsize=7.5, color='#52514e')
    fig.subplots_adjust(left=0.085, right=0.98, top=0.85, bottom=0.32, wspace=0.20)
    fig.savefig(out_png, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_table(stats, args, out_csv):
    import csv
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'backbone', 'num_experts', 'mean_avg_width', 'std_avg_width',
                    'n_seeds'] + [f'seed_{s}' for s in args.seeds])
        for spec in SERIES:
            for ne in args.experts:
                st = stats.get((spec['key'], ne))
                if not st:
                    continue
                w.writerow([spec['display'], VARIANT_NAME.get(spec['variant'], '?'),
                            ne, f"{st['mean']:.6f}",
                            f"{st['std']:.6f}" if st['n'] > 1 else '',
                            st['n']] + [f"{st['per_seed'][s]:.6f}" if s in st['per_seed'] else ''
                                        for s in args.seeds])


def report(picked, stats, args):
    print(f'\n{args.dataset} / {args.model} / pred_len {args.pred_len} — mean avg width '
          f'(± s.d. over {len(args.seeds)} seeds)')
    width = max(len(s['display']) for s in SERIES) + 6
    for spec in SERIES:
        cells = []
        for ne in args.experts:
            st = stats.get((spec['key'], ne))
            cells.append(f'{st["mean"]:.3f}±{st["std"]:.3f}({st["n"]})' if st else '     --     ')
        print(f'  {labelled(spec):>{width}}  ' + '  '.join(cells))
    print('  ' + ' ' * width + '  ' + '  '.join(f'{"ne=" + str(ne):^13}' for ne in args.experts))

    missing = [(spec['display'], ne, s) for spec in SERIES for ne in args.experts
               for s in args.seeds if (spec['key'], ne, s) not in picked]
    if missing:
        by_method = collections.defaultdict(list)
        for name, ne, s in missing:
            by_method[name].append(f'ne{ne}/seed{s}')
        print(f'\n  {len(missing)} of {len(SERIES) * len(args.experts) * len(args.seeds)} '
              f'(method, experts, seed) cells have no result yet:')
        for name, cells in by_method.items():
            shown = ', '.join(cells[:8]) + (f' … (+{len(cells) - 8})' if len(cells) > 8 else '')
            print(f'    {name}: {shown}')

    disagree = {k: v for k, v in picked.items() if v['spread'] > 5e-4}
    if disagree:
        print(f'\n  {len(disagree)} cell(s) were observed more than once with differing '
              f'widths (model re-trained); the result_*.txt value is used:')
        for (key, ne, seed), v in sorted(disagree.items()):
            print(f'    {key} ne{ne} seed{seed}: spread {v["spread"]:.4f} '
                  f'over {v["n_obs"]} observations -> {v["width"]:.4f} ({v["source"]})')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', default='ETTh1')
    p.add_argument('--model', default='iTransformer')
    p.add_argument('--pred_len', type=int, default=96)
    p.add_argument('--experts', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    p.add_argument('--seeds', type=int, nargs='+', default=[4021, 4022, 4023, 4024, 4025])
    p.add_argument('--window', type=int, default=DEFAULT_WINDOW,
                   help='calibration sliding-window size to keep (default 1000)')
    p.add_argument('--cp_variant', choices=list(VARIANT_CODE), default='MoE',
                   help='backbone the plain-CP line is read off (default MoE). MoECP and '
                        'AleatoricScaleCP need the mixture-density head, so they stay on MoG.')
    p.add_argument('--outdir', default=os.path.join(ROOT, 'figures'))
    args = p.parse_args()
    spec_of('standard_cp')['variant'] = VARIANT_CODE[args.cp_variant]

    os.makedirs(args.outdir, exist_ok=True)
    stem = f'width_vs_experts_{args.dataset}_{args.model}_pl{args.pred_len}'
    out_png = os.path.join(args.outdir, stem + '.png')
    out_csv = os.path.join(args.outdir, stem + '.csv')

    picked = collect(args)
    stats = aggregate(picked, args)
    if not stats:
        raise SystemExit('No matching calibration results found -- nothing to plot.')
    plot(stats, args, out_png)
    write_table(stats, args, out_csv)
    report(picked, stats, args)
    print(f'\nwrote {os.path.relpath(out_png, ROOT)} (300 dpi)'
          f'\nwrote {os.path.relpath(out_csv, ROOT)}')


if __name__ == '__main__':
    main()
