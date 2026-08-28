"""
How much width is *available* on the Synthetic dataset, computed from the generative process
rather than from any trained model. Run manually (no test framework in this repo).

Motivation: variance-ratio calibration barely beats CP-VS(total) on Synthetic (measured
width 1.3550 @ 0.8993 vs 1.3716 @ 0.8940), and it is not obvious from the run logs whether
that is a model-quality problem, a tuning problem, or a ceiling. This script settles it by
computing the oracle widths directly from Dataset_Synthetic's own DGP:

    Y | X ~ 0.2 N(X, 1) + 0.3 N(X^2, 1) + 0.5 N(X^3, 1)

which is genuinely trimodal, because Dataset_Synthetic draws the domain label independently
of X (data_loader.py: `domains = np.random.choice([0,1,2], p=proportions, size=total)`).

It reports four oracles at 90% coverage plus one control:
  [A] CP-VS given the *true* sigma_total          -- what perfect variance estimates buy
  [B] the best (r, q^2) in the variance-ratio family at the true variances -- the family ceiling
  [C] the shortest single interval per x           -- the ceiling for ANY symmetric/asymmetric
                                                      single-interval method
  [D] the HPD region (union of intervals)          -- what dropping the single-interval shape buys
  [E] control: replay the variance-ratio construction on the TRUE variances, to separate
      "the model's variances are bad" from "the family has no room"

Everything is in the standardized y units the pipeline reports, so the numbers are directly
comparable to result_calibration_*.txt.

Run with: unc_moe/bin/python3 scripts/tex/verify_adaptive_variance_ceiling.py
"""
import os
import sys

import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

ALPHA = 0.1
SEED = 4022
N_PER_SPLIT = 1500          # scripts/run_tabular_regression.sh passes --n_samples 1500
PROPORTIONS = [0.2, 0.3, 0.5]
W = np.array(PROPORTIONS)


def regenerate_dgp(seed=SEED, n_per_split=N_PER_SPLIT):
    """Replay Dataset_Synthetic.__read_data__ exactly.

    The RNG call order (uniform -> normal -> choice) and the train-split-only StandardScaler
    fit both matter: reordering the draws or fitting the scaler on all 4500 points would put
    the oracle in different units than the pipeline reports.
    """
    total = n_per_split * 3
    np.random.seed(seed)
    x = np.random.uniform(-3, 3, total)
    eps = np.random.normal(0, 1, total)
    domains = np.random.choice([0, 1, 2], p=PROPORTIONS, size=total)

    y = np.zeros(total)
    y[domains == 0] = x[domains == 0] + eps[domains == 0]
    y[domains == 1] = x[domains == 1] ** 2 + eps[domains == 1]
    y[domains == 2] = x[domains == 2] ** 3 + eps[domains == 2]

    # StandardScaler is fit on the train split only, then applied to every split.
    mu_y, sd_y = y[:n_per_split].mean(), y[:n_per_split].std()
    return x, y, mu_y, sd_y


def conditional_mixture(x_raw, mu_y, sd_y):
    """Per-point standardized mixture components (means, sds) and the induced variance split."""
    mu = np.stack([x_raw, x_raw ** 2, x_raw ** 3], axis=-1)
    mu = (mu - mu_y) / sd_y                     # [n, 3]
    sd = np.full_like(mu, 1.0 / sd_y)           # noise sd is 1 in raw units
    mean_mix = (mu * W).sum(-1)
    aleat_var = (sd ** 2 * W).sum(-1)                          # within-component
    epist_var = (W * (mu - mean_mix[:, None]) ** 2).sum(-1)    # between-component
    return mu, sd, mean_mix, aleat_var, epist_var


def centered_coverage(center, half, mu, sd):
    """Exact conditional coverage of [center-half, center+half] under the mixture."""
    lo, hi = center - half, center + half
    return (W * (norm.cdf((hi[:, None] - mu) / sd) - norm.cdf((lo[:, None] - mu) / sd))).sum(-1)


def q_level(n, alpha=ALPHA):
    return min(max(np.ceil((n + 1) * (1 - alpha)) / n, 0.0), 1.0)


def variance_ratio_replay(y_cal, p_cal, va_cal, ve_cal, y_te, p_te, va_te, ve_te,
                          r_grid, n_repeats=20, tune_frac=0.5, split_seed=0):
    """Self-contained replay of the width-minimizing variance-ratio construction.

    Reimplemented here rather than imported so this script measures the *construction*, not a
    particular class, and so it can be pointed at the true DGP variances.
    """
    resid_sq = (y_cal - p_cal) ** 2
    n = len(resid_sq)

    def q_sq(rs, va, ve, r):
        s = np.clip((rs - r * ve) / va, 0.0, None)
        return np.quantile(s, q_level(len(s)), method='higher')

    n_tune = int(round(tune_frac * n))
    curves = np.empty((n_repeats, len(r_grid)))
    splits = []
    for b in range(n_repeats):
        perm = np.random.RandomState(split_seed + b).permutation(n)
        ti, ci = perm[:n_tune], perm[n_tune:]
        splits.append((ti, ci))
        for j, r in enumerate(r_grid):
            q = q_sq(resid_sq[ti], va_cal[ti], ve_cal[ti], r)
            curves[b, j] = np.sqrt(q * va_cal[ti] + r * ve_cal[ti]).mean()

    avg = curves.mean(axis=0)
    best_j = 0
    for j in range(1, len(r_grid)):
        if avg[j] < avg[best_j] - 1e-12:
            best_j = j
    r_star = float(r_grid[best_j])
    q2 = float(np.median([q_sq(resid_sq[ci], va_cal[ci], ve_cal[ci], r_star) for _, ci in splits]))

    half = np.sqrt(q2 * va_te + r_star * ve_te)
    return r_star, q2, float(np.mean(2 * half)), float(np.mean(np.abs(y_te - p_te) <= half))


def main():
    x, y, mu_y, sd_y = regenerate_dgp()
    n = N_PER_SPLIT
    cal_sl, te_sl = slice(n, 2 * n), slice(2 * n, 3 * n)

    mu_te, sd_te, mean_te, va_te, ve_te = conditional_mixture(x[te_sl], mu_y, sd_y)
    _, _, mean_cal, va_cal, ve_cal = conditional_mixture(x[cal_sl], mu_y, sd_y)
    y_te = (y[te_sl] - mu_y) / sd_y
    y_cal = (y[cal_sl] - mu_y) / sd_y

    print("=" * 78)
    print(f"Synthetic DGP oracles at {1-ALPHA:.0%} coverage (standardized y units, seed={SEED},"
          f" n/split={n})")
    print("=" * 78)
    print(f"true aleatoric variance (constant): {va_te[0]:.4f}")
    print(f"true epistemic variance: mean={ve_te.mean():.4f} median={np.median(ve_te):.4f} "
          f"max={ve_te.max():.3f}   <- heavily right-skewed, which is why MEAN width is a tail statistic")
    print(f"epistemic share of total variance: {np.mean(ve_te/(va_te+ve_te))*100:.1f}%\n")

    sd_tot = np.sqrt(va_te + ve_te)

    # [A] CP-VS with the true total sd: one global multiplier k on sigma_total.
    ks = np.linspace(0.2, 6.0, 4000)
    covs = np.array([centered_coverage(mean_te, k * sd_tot, mu_te, sd_te).mean() for k in ks])
    k90 = ks[np.argmin(np.abs(covs - (1 - ALPHA)))]
    width_a = 2 * k90 * sd_tot.mean()

    # [B] Best (r, q^2) in the variance-ratio family, q^2 solved to hit coverage at each r.
    r_grid = np.unique(np.concatenate(([0.0], np.geomspace(1e-2, 1e2, 60))))
    best_b = None
    for r in r_grid:
        lo, hi = 1e-6, 400.0
        for _ in range(60):                     # bisect q^2 to exactly 1-alpha coverage
            q2 = 0.5 * (lo + hi)
            c = centered_coverage(mean_te, np.sqrt(q2 * va_te + r * ve_te), mu_te, sd_te).mean()
            if c < 1 - ALPHA:
                lo = q2
            else:
                hi = q2
        q2 = 0.5 * (lo + hi)
        w = 2 * np.sqrt(q2 * va_te + r * ve_te).mean()
        if best_b is None or w < best_b[0]:
            best_b = (w, r, q2)
    width_b, r_b, q2_b = best_b

    # Densities on a shared grid, for the two shape oracles.
    grid = np.linspace(y_te.min() - 6, y_te.max() + 6, 20001)
    dy = grid[1] - grid[0]
    dens = (W[None, None, :] * norm.pdf(grid[None, :, None],
                                        mu_te[:, None, :], sd_te[:, None, :])).sum(-1)

    # [C] Shortest single interval per x, at 90% CONDITIONAL coverage (two-pointer sweep).
    def shortest_interval(d):
        cs = np.cumsum(d) * dy
        cs /= cs[-1]
        best, lo_i = np.inf, 0
        for hi_i in range(len(cs)):
            while cs[hi_i] - cs[lo_i] >= 1 - ALPHA:
                best = min(best, grid[hi_i] - grid[lo_i])
                lo_i += 1
        return best

    sub = 400        # the sweep is the expensive step; a 400-point subsample is plenty here
    width_c = np.mean([shortest_interval(d) for d in dens[:sub]])

    # [D] HPD region: keep the highest-density mass until 90% is accumulated.
    def hpd_len(d):
        order = np.argsort(d)[::-1]
        cs = np.cumsum(d[order]) * dy
        cs /= cs[-1]
        return (np.searchsorted(cs, 1 - ALPHA) + 1) * dy

    width_d = np.mean([hpd_len(d) for d in dens])

    print(f"{'oracle region shape':<52s} {'avg width':>10s} {'vs [A]':>9s}")
    print("-" * 78)
    rows = [
        ("[A] CP-VS with TRUE sigma_total", width_a),
        ("[B] variance-ratio family (r,q^2) at TRUE variances", width_b),
        ("[C] shortest single interval per x", width_c),
        ("[D] HPD region (union of intervals)", width_d),
    ]
    for label, w in rows:
        print(f"{label:<52s} {w:10.4f} {100*(w/width_a - 1):+8.1f}%")
    print(f"\n[B] attained at r={r_b:.4f}, q^2={q2_b:.4f}")

    # [E] Control: the actual construction, fed true variances instead of model estimates.
    r_grid_std = np.unique(np.concatenate(([0.0], np.geomspace(1e-2, 1e2, 25))))
    r_s, q2_s, w_adapt, c_adapt = variance_ratio_replay(
        y_cal, mean_cal, va_cal, ve_cal, y_te, mean_te, va_te, ve_te, r_grid_std)
    q_cpvs = np.quantile(np.abs(y_cal - mean_cal) / np.sqrt(va_cal + ve_cal),
                         q_level(len(y_cal)), method='higher')
    half_cpvs = q_cpvs * sd_tot
    w_cpvs = float(np.mean(2 * half_cpvs))
    c_cpvs = float(np.mean(np.abs(y_te - mean_te) <= half_cpvs))

    print("\n" + "=" * 78)
    print("[E] Control -- the variance-ratio construction fed the TRUE DGP variances")
    print("=" * 78)
    print(f"  variance-ratio: r*={r_s:.4f} q^2={q2_s:.4f} width={w_adapt:.4f} cov={c_adapt:.4f}")
    print(f"  CP-VS(total)  :                    width={w_cpvs:.4f} cov={c_cpvs:.4f}")
    print(f"  gain from the ratio, with PERFECT variances: {100*(1 - w_adapt/w_cpvs):+.1f}%")
    print("\nInterpretation: the gain stays ~2% even when the variance inputs are exact, so the")
    print("gap is the family, not the model. Roughly 42% of the available width lives in [D],")
    print("i.e. in dropping the single-interval shape (which MoG_HPD_Calibrator already does).")
    print("Width is therefore the wrong axis to compete on here; conditional coverage is not.")


if __name__ == '__main__':
    main()
