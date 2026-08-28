"""
Standalone sanity checks for AdaptiveVarianceCalibrator's repeated/cross-fitted tune-calib
splitting, run manually (no test framework in this repo). Uses hand-constructed
resid/aleat_var/epist_var (not a trained model) to verify:
  1. n_repeats=1 reproduces the pre-change single-split algorithm exactly (same RandomState(seed)
     draw, same tie-break rule, q_sq = the one calib fold's quantile) -- confirms the refactor is a
     strict superset, not a behavior change, when repeats are turned off.
  2. Averaging over many repeats materially reduces the spread of the chosen r (and q_sq) across
     different `split_seed` values, at a *fixed* calibration set -- this is the instability observed
     in result_calibration_adaptive_variance.txt across reruns of the same trained checkpoint.
  3. Marginal coverage stays ~90% with the new default, confirming the stabilization doesn't change
     the validity properties the existing diagnostics already check.

Run with: unc_moe/bin/python3 scripts/tex/verify_adaptive_variance_stability.py
"""
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from calibration.tabular_regression.Adaptive_Variance_Calibrator import AdaptiveVarianceCalibrator


def make_synthetic_calibration_set(rng, n, true_r=1.5):
    """resid^2 ~= aleat_var + true_r * epist_var, at a scale similar to the real tabular datasets
    (Bike/Temperature/Synthetic calibration sets are ~1500-3400 points)."""
    aleat_var = rng.uniform(0.5, 2.0, size=n)
    epist_var = rng.uniform(0.0, 2.0, size=n)
    resid = rng.normal(0.0, np.sqrt(aleat_var + true_r * epist_var))

    cal_preds = torch.zeros(n)
    cal_trues = torch.tensor(resid, dtype=torch.float32)
    cal_stds_aleat = torch.tensor(np.sqrt(aleat_var), dtype=torch.float32)
    cal_stds_epist = torch.tensor(np.sqrt(epist_var), dtype=torch.float32)
    return cal_preds, cal_trues, cal_stds_aleat, cal_stds_epist, aleat_var, epist_var


def old_single_split_fit(resid_sq, aleat_var, epist_var, r_grid, alpha, tune_frac, split_seed):
    """Direct reimplementation of the pre-change single-split algorithm, used only as an oracle
    to confirm n_repeats=1 matches it exactly."""
    n_total = len(resid_sq)
    rng = np.random.RandomState(split_seed)
    perm = rng.permutation(n_total)
    n_tune = int(round(tune_frac * n_total))
    tune_idx, calib_idx = perm[:n_tune], perm[n_tune:]

    def q_sq_for(rs, av, ev, r):
        scores = np.clip((rs - r * ev) / av, 0.0, None)
        q_level = min(max(np.ceil((len(scores) + 1) * (1 - alpha)) / len(scores), 0.0), 1.0)
        return np.quantile(scores, q_level, method='higher')

    best_r, best_width = None, np.inf
    for r in r_grid:
        q_sq_tune = q_sq_for(resid_sq[tune_idx], aleat_var[tune_idx], epist_var[tune_idx] + 0, r)
        width_r = np.sqrt(q_sq_tune * aleat_var[tune_idx] + r * epist_var[tune_idx]).mean()
        if width_r < best_width - 1e-12:
            best_width, best_r = width_r, r

    q_sq = q_sq_for(resid_sq[calib_idx], aleat_var[calib_idx], epist_var[calib_idx], best_r)
    return float(best_r), float(q_sq)


def check_n_repeats_1_matches_old_algorithm():
    print("=== Check 1: n_repeats=1 reproduces the old single-split algorithm exactly ===")
    rng = np.random.RandomState(0)
    n = 2000
    cal_preds, cal_trues, cal_stds_aleat, cal_stds_epist, aleat_var, epist_var = \
        make_synthetic_calibration_set(rng, n)
    resid_sq = (cal_trues.numpy() - cal_preds.numpy()) ** 2
    aleat_var_eps = aleat_var + 1e-8  # matches the +1e-8 fit() adds internally

    for split_seed in [0, 3, 7]:
        calib = AdaptiveVarianceCalibrator(alpha=0.1, split_seed=split_seed, n_repeats=1)
        calib.fit(cal_preds, cal_trues, cal_stds_aleat, cal_stds_epist)

        oracle_r, oracle_q_sq = old_single_split_fit(
            resid_sq, aleat_var_eps, epist_var, calib.r_grid, 0.1, 0.5, split_seed)

        print(f"  split_seed={split_seed}: new r={calib.r:.6f} vs oracle r={oracle_r:.6f}, "
              f"new q_sq={calib.q_sq:.6f} vs oracle q_sq={oracle_q_sq:.6f}")
        # float32 (calibrator, via torch) vs float64 (numpy oracle) introduces tiny rounding
        # differences; tolerance is loose enough to absorb that while still confirming the same
        # algorithm/tune-calib split is being run.
        assert abs(calib.r - oracle_r) < 1e-4, "n_repeats=1 should match the old algorithm's r"
        assert abs(calib.q_sq - oracle_q_sq) < 1e-4, "n_repeats=1 should match the old algorithm's q_sq"
    print("  PASS\n")


def check_repeats_reduce_split_seed_variance():
    print("=== Check 2: averaging over repeats reduces r/q_sq spread across split_seed draws ===")
    rng = np.random.RandomState(1)
    n = 2000
    cal_preds, cal_trues, cal_stds_aleat, cal_stds_epist, _, _ = \
        make_synthetic_calibration_set(rng, n, true_r=1.5)

    split_seeds = list(range(30))
    for n_repeats, label in [(1, "old (n_repeats=1)"), (20, "new default (n_repeats=20)")]:
        rs, q_sqs = [], []
        for split_seed in split_seeds:
            calib = AdaptiveVarianceCalibrator(alpha=0.1, split_seed=split_seed, n_repeats=n_repeats)
            calib.fit(cal_preds, cal_trues, cal_stds_aleat, cal_stds_epist)
            rs.append(calib.r)
            q_sqs.append(calib.q_sq)
        r_std, q_sq_std = np.std(rs), np.std(q_sqs)
        print(f"  {label}: r std over {len(split_seeds)} split_seeds = {r_std:.4f} "
              f"(mean r={np.mean(rs):.4f}), q_sq std = {q_sq_std:.4f}")
        if n_repeats == 1:
            old_r_std, old_q_sq_std = r_std, q_sq_std
        else:
            new_r_std, new_q_sq_std = r_std, q_sq_std

    print(f"  reduction: r std {old_r_std:.4f} -> {new_r_std:.4f}, "
          f"q_sq std {old_q_sq_std:.4f} -> {new_q_sq_std:.4f}")
    assert new_r_std < 0.5 * old_r_std, "repeated splitting should materially reduce r instability"
    assert new_q_sq_std < 0.5 * old_q_sq_std, "repeated splitting should materially reduce q_sq instability"
    print("  PASS\n")


def check_marginal_coverage():
    print("=== Check 3: marginal coverage stays ~90% with the new default (n_repeats=20) ===")
    rng = np.random.RandomState(2)
    n_cal, n_test = 3000, 5000
    cal_preds, cal_trues, cal_stds_aleat, cal_stds_epist, _, _ = \
        make_synthetic_calibration_set(rng, n_cal, true_r=1.5)
    test_preds, test_trues, test_stds_aleat, test_stds_epist, _, _ = \
        make_synthetic_calibration_set(rng, n_test, true_r=1.5)

    calib = AdaptiveVarianceCalibrator(alpha=0.1)  # new default n_repeats=20
    calib.fit(cal_preds, cal_trues, cal_stds_aleat, cal_stds_epist)
    intervals = calib.predict(test_preds, test_stds_aleat, test_stds_epist).numpy()

    covered = (test_trues.numpy() >= intervals[:, 0]) & (test_trues.numpy() <= intervals[:, 1])
    coverage = covered.mean()
    print(f"  r={calib.r:.4f}, q_sq={calib.q_sq:.4f}, n_repeats_={calib.n_repeats_}, "
          f"coverage={coverage:.4f} (target ~0.90)")
    assert 0.87 <= coverage <= 0.93, f"coverage {coverage} outside expected band"
    print("  PASS\n")


if __name__ == "__main__":
    check_n_repeats_1_matches_old_algorithm()
    check_repeats_reduce_split_seed_variance()
    check_marginal_coverage()
    print("All AdaptiveVarianceCalibrator stability checks passed.")
