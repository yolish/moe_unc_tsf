"""
Standalone sanity checks for MoG_HPD_Calibrator, run manually (no test framework in this repo).
Uses hand-constructed mu/sigma/pi (not the trained model) to verify:
  1. K=1 reduces to a single interval symmetric about mu, with coverage/width converging to the
     Gaussian's known 90% analytic values.
  2. Two well-separated narrow components produce two disjoint segments, far narrower than a
     single interval spanning both modes.
  3. Coincident modes never spuriously split into more than one segment.
  4. Marginal coverage is ~90% on a held-out set from a known generative mixture, regardless of
     whether the calibration and test mu/sigma/pi match the true generative parameters exactly.

Run with: unc_moe/bin/python3 scripts/tex/verify_mog_hpd.py
"""
import numpy as np
from scipy.stats import norm

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from calibration.tabular_regression.MoG_HPD_Calibrator import MoG_HPD_Calibrator


def check_k1_reduction():
    print("=== Check 1: K=1 reduces to a single symmetric interval ===")
    rng = np.random.RandomState(0)
    mu_true, sigma_true = 2.0, 1.5
    n_cal = 20000
    y_cal = rng.normal(mu_true, sigma_true, size=n_cal)

    mu = np.full((n_cal, 1), mu_true)
    sigma = np.full((n_cal, 1), sigma_true)
    pi = np.ones((n_cal, 1))

    calib = MoG_HPD_Calibrator(alpha=0.1)
    calib.fit(y_cal, mu, sigma, pi)

    n_test = 5000
    y_test = rng.normal(mu_true, sigma_true, size=n_test)
    test_mu = np.full((n_test, 1), mu_true)
    test_sigma = np.full((n_test, 1), sigma_true)
    test_pi = np.ones((n_test, 1))

    intervals = calib.predict(test_mu, test_sigma, test_pi)

    n_segs = np.array([len(seg) for seg in intervals])
    assert np.all(n_segs == 1), f"expected exactly 1 segment for every point, got counts {np.unique(n_segs)}"

    widths = np.array([seg[0, 1] - seg[0, 0] for seg in intervals])
    centers = np.array([(seg[0, 1] + seg[0, 0]) / 2 for seg in intervals])
    assert np.allclose(centers, mu_true, atol=1e-3), "interval should be symmetric about mu"

    covered = np.array([bool(seg[0, 0] <= y <= seg[0, 1]) for y, seg in zip(y_test, intervals)])
    coverage = covered.mean()
    avg_width = widths.mean()
    analytic_width = 2 * sigma_true * norm.ppf(0.95)  # 90% analytic HPD width for a Gaussian

    print(f"  coverage = {coverage:.4f} (target ~0.90)")
    print(f"  avg width = {avg_width:.4f} vs analytic 90% width = {analytic_width:.4f}")
    assert 0.88 <= coverage <= 0.92, f"coverage {coverage} outside expected band"
    assert abs(avg_width - analytic_width) / analytic_width < 0.05, "width should track the analytic Gaussian HPD width"
    print("  PASS\n")


def check_two_separated_components():
    print("=== Check 2: two well-separated narrow components give 2 disjoint segments ===")
    rng = np.random.RandomState(1)
    n_cal = 20000
    mode = rng.binomial(1, 0.5, size=n_cal)
    mus_true = np.where(mode == 0, -5.0, 5.0)
    sigma_true = 0.3
    y_cal = rng.normal(mus_true, sigma_true)

    mu = np.stack([np.full(n_cal, -5.0), np.full(n_cal, 5.0)], axis=1)
    sigma = np.full((n_cal, 2), sigma_true)
    pi = np.full((n_cal, 2), 0.5)

    calib = MoG_HPD_Calibrator(alpha=0.1)
    calib.fit(y_cal, mu, sigma, pi)

    test_mu = mu[:1000]
    test_sigma = sigma[:1000]
    test_pi = pi[:1000]
    intervals = calib.predict(test_mu, test_sigma, test_pi)

    n_segs = np.array([len(seg) for seg in intervals])
    frac_two = np.mean(n_segs == 2)
    print(f"  fraction of points with exactly 2 segments: {frac_two:.4f}")
    assert frac_two > 0.95, f"expected the vast majority of points to split into 2 segments, got {frac_two}"

    widths = np.array([seg[:, 1].sum() - seg[:, 0].sum() for seg in intervals if len(seg) == 2])
    avg_width = widths.mean()
    single_component_width = 2 * sigma_true * norm.ppf(0.95)
    naive_span = (5.0 + 6 * sigma_true) - (-5.0 - 6 * sigma_true)  # a single interval spanning both modes

    print(f"  avg total width (2 segments) = {avg_width:.4f}")
    print(f"  ~2x single-component 90% width = {2 * single_component_width:.4f}")
    print(f"  naive single-interval span (for comparison) = {naive_span:.4f}")
    assert avg_width < naive_span / 2, "disjoint segments should be far narrower than one spanning interval"
    print("  PASS\n")


def check_coincident_modes():
    print("=== Check 3: coincident modes never spuriously split ===")
    rng = np.random.RandomState(2)
    n_cal = 5000
    mu = np.zeros((n_cal, 3))
    sigma = np.stack([np.full(n_cal, 0.5), np.full(n_cal, 1.0), np.full(n_cal, 1.5)], axis=1)
    pi = np.full((n_cal, 3), 1.0 / 3)

    comp = rng.choice(3, size=n_cal, p=[1 / 3, 1 / 3, 1 / 3])
    y_cal = rng.normal(0.0, sigma[np.arange(n_cal), comp])

    calib = MoG_HPD_Calibrator(alpha=0.1)
    calib.fit(y_cal, mu, sigma, pi)

    intervals = calib.predict(mu[:500], sigma[:500], pi[:500])
    n_segs = np.array([len(seg) for seg in intervals])
    print(f"  segment counts observed: {np.unique(n_segs)}")
    assert np.all(n_segs <= 1), f"coincident modes should never split, got counts {np.unique(n_segs)}"
    print("  PASS\n")


def check_marginal_coverage_model_agnostic():
    print("=== Check 4: marginal coverage ~90% under a misspecified mixture fit ===")
    rng = np.random.RandomState(3)
    n_cal, n_test = 20000, 20000
    mode = rng.binomial(1, 0.4, size=n_cal + n_test)
    true_mu = np.where(mode == 0, -2.0, 3.0)
    true_sigma = np.where(mode == 0, 0.8, 1.2)
    y = rng.normal(true_mu, true_sigma)

    # Deliberately mis-specified fitted mixture (weights/means/stds slightly off the true ones)
    # to confirm coverage validity doesn't depend on model correctness, only on the split-
    # conformal quantile step.
    fitted_mu = np.stack([np.full(n_cal + n_test, -1.7), np.full(n_cal + n_test, 3.3)], axis=1)
    fitted_sigma = np.stack([np.full(n_cal + n_test, 1.0), np.full(n_cal + n_test, 1.4)], axis=1)
    fitted_pi = np.full((n_cal + n_test, 2), 0.5)

    y_cal, y_test = y[:n_cal], y[n_cal:]
    mu_cal, mu_test = fitted_mu[:n_cal], fitted_mu[n_cal:]
    sigma_cal, sigma_test = fitted_sigma[:n_cal], fitted_sigma[n_cal:]
    pi_cal, pi_test = fitted_pi[:n_cal], fitted_pi[n_cal:]

    calib = MoG_HPD_Calibrator(alpha=0.1)
    calib.fit(y_cal, mu_cal, sigma_cal, pi_cal)
    intervals = calib.predict(mu_test, sigma_test, pi_test)

    covered = np.array([bool(len(seg)) and bool(np.any((seg[:, 0] <= yy) & (yy <= seg[:, 1])))
                         for yy, seg in zip(y_test, intervals)])
    coverage = covered.mean()
    print(f"  coverage = {coverage:.4f} (target ~0.90, model is deliberately mis-specified)")
    assert 0.87 <= coverage <= 0.93, f"coverage {coverage} outside expected band"
    print("  PASS\n")


if __name__ == "__main__":
    check_k1_reduction()
    check_two_separated_components()
    check_coincident_modes()
    check_marginal_coverage_model_agnostic()
    print("All MoG-HPD sanity checks passed.")
