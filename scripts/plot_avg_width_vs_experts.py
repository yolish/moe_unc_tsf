"""Average interval width vs. number of experts for ETTh1 -- one ICASSP-ready figure.

All five methods on a single axes: one line each, distinct marker and dash pattern,
error bars at ± 1 s.d. across the five target seeds. Fixed to the requested cell:
ETTh1 / iTransformer / pred_len 96, experts 1-5, seeds 4021-4025, sliding window 1000.

    python scripts/plot_avg_width_vs_experts.py

Writes figures/avg_width_vs_experts_ETTh1.{pdf,png} -- PDF for LaTeX (vector, no
resampling at any zoom), PNG at 300 dpi for slides and quick viewing.

Styled to IEEE/ICASSP conventions: 3.5 in single-column width, Times-family serif at
paper body size, boxed axes with inward ticks, legend above the frame. The figure
carries no title and no description block -- both belong in the paper's \\caption, and
duplicating them on the page is what an ICASSP reviewer notices first. A ready caption
is printed when the script runs; the full figure block with the surrounding paragraph
lives in docs/fig_avg_width_vs_experts.tex.

Record parsing (result_calibration_*.txt + logs/**/*.log, txt beating log for the same
cell) is shared with plot_width_vs_experts.py, which renders the same numbers with a
second zoom panel for the CP-family error bars.
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plot_width_vs_experts import ROOT, SERIES, VARIANT_NAME, aggregate, collect

DATASET, MODEL, PRED_LEN = 'ETTh1', 'iTransformer', 96
EXPERTS = [1, 2, 3, 4, 5]
SEEDS = [4021, 4022, 4023, 4024, 4025]
WINDOW = 1000

# IEEE/ICASSP figure conventions. Nimbus Roman is the URW Times clone and is what the
# box has; the rest of the list is fallback. Sizes are set against a 3.5 in single
# column so the type in the figure matches the paper's 9 pt body text once placed.
ICASSP_RC = {
    'font.family': 'serif',
    'font.serif': ['Nimbus Roman', 'Liberation Serif', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.linewidth': 0.6,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'legend.fontsize': 7.0,
    'legend.handlelength': 2.4,
    'legend.borderpad': 0.4,
    'legend.labelspacing': 0.3,
    'legend.columnspacing': 1.0,
    'lines.linewidth': 1.1,
    'lines.markersize': 3.6,
    'grid.linewidth': 0.4,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'pdf.fonttype': 42,   # embed TrueType, not Type 3 -- IEEE PDF-eXpress rejects Type 3
    'ps.fonttype': 42,
}
FIGSIZE = (3.5, 2.45)   # IEEE single column
ALPHA = 0.1             # every calibrator here is built with alpha=0.1 -> 90% intervals

# Legend text for the chart. The aleatoric-scale method is named CP-MOG, and the two
# CQR configurations are spelled out so they are not read as one series.
DISPLAY = {
    'standard_cp': 'CP',
    'cqr_quantile': 'CQR (standard)',
    'cqr_retrain': 'CQR (rolling retrain)',
    'moecp': 'MoECP',
    'aleatoric_scale': 'CP-MOG',
}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--outdir', default=os.path.join(ROOT, 'figures'))
    args = p.parse_args()
    args.dataset, args.model, args.pred_len = DATASET, MODEL, PRED_LEN
    args.experts, args.seeds, args.window = EXPERTS, SEEDS, WINDOW

    stats = aggregate(collect(args), args)
    if not stats:
        raise SystemExit('No matching calibration results found -- nothing to plot.')

    plt.rcParams.update(ICASSP_RC)
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for spec in SERIES:
        xs = [ne for ne in EXPERTS if (spec['key'], ne) in stats]
        if not xs:
            print(f"  warning: no data for {DISPLAY[spec['key']]} -- omitted")
            continue
        ax.errorbar(xs,
                    [stats[(spec['key'], ne)]['mean'] for ne in xs],
                    yerr=[stats[(spec['key'], ne)]['std'] for ne in xs],
                    capsize=2.5, capthick=0.8, elinewidth=0.8,
                    color=spec['color'], marker=spec['marker'],
                    markeredgecolor='white', markeredgewidth=0.4,
                    linestyle=spec['ls'],
                    label=DISPLAY[spec['key']], zorder=3)

    ax.set_xlabel('Number of experts')
    ax.set_ylabel('Avg. interval width')
    ax.set_xticks(EXPERTS)
    ax.set_xlim(min(EXPERTS) - 0.2, max(EXPERTS) + 0.2)
    # Headroom above the data for the legend. Parking it in the empty mid-band reads as
    # floating; the conventional place is the top of the frame, which needs room made
    # for it. 0.45 of the data range clears the tallest error bar at two columns.
    top = max(stats[k]['mean'] + stats[k]['std'] for k in stats)
    bot = min(stats[k]['mean'] - stats[k]['std'] for k in stats)
    ax.set_ylim(bot - 0.04 * (top - bot), top + 0.45 * (top - bot))
    ax.grid(True, axis='y', color='0.85', linestyle='-')
    ax.set_axisbelow(True)

    # Two columns keep the legend to three rows; five rows would need twice the headroom.
    leg = ax.legend(loc='upper center', ncol=2, frameon=True, framealpha=1.0,
                    edgecolor='0.7', fancybox=False, fontsize=6.5, borderpad=0.35,
                    labelspacing=0.25, handlelength=2.0, columnspacing=0.8,
                    handletextpad=0.5)
    leg.get_frame().set_linewidth(0.5)
    leg.set_zorder(5)

    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.join(args.outdir, f'avg_width_vs_experts_{DATASET}')
    fig.savefig(stem + '.pdf')
    fig.savefig(stem + '.png', dpi=300)
    plt.close(fig)

    for spec in SERIES:
        row = [f'{stats[(spec["key"], ne)]["mean"]:.3f}±{stats[(spec["key"], ne)]["std"]:.3f}'
               if (spec['key'], ne) in stats else '   --   ' for ne in EXPERTS]
        print(f'  {DISPLAY[spec["key"]]:>22}  ' + '  '.join(row))
    print(f'\nwrote {os.path.relpath(stem, ROOT)}.pdf (vector, for LaTeX)'
          f'\nwrote {os.path.relpath(stem, ROOT)}.png (300 dpi)')
    print(f'\nsuggested caption (full block: docs/fig_avg_width_vs_experts.tex):\n'
          f'  \\caption{{Average width of the ${int((1 - ALPHA) * 100)}\\%$ prediction '
          f'intervals on {DATASET} ({MODEL}, horizon ${PRED_LEN}$) vs.\\ the number of '
          f'experts; mean $\\pm 1$ s.d.\\ over {len(SEEDS)} seeds.}}')


if __name__ == '__main__':
    main()
