import numpy as np
from collections import Counter

from calibration.tabular_regression.MoG_HPD_Calibrator import MoG_HPD_Calibrator


class STAHPDCalibrator(MoG_HPD_Calibrator):
    """
    Shape-Threshold Adaptive HPD conformal calibration for Mixture-of-Gaussians models.

    Motivation
    ----------
    MoG_HPD_Calibrator conformalizes S = -log f_mix(y|x) and returns the global super-level set
    {y : f_mix(y|x) >= tau_hat}. By the Neyman-Pearson argument in docs/calibration_methods.tex
    that is the minimum-measure set achieving a given coverage *under the density it is handed*,
    and thresholding at one global tau is the correct functional form of the width-optimal
    marginal rule -- but only when the model density equals the truth. Measured on this repo's
    four tabular datasets, two things follow:

      * Re-leveling alone is exhausted. An oracle per-sigma-decile offset to tau_hat, fitted
        directly on the test set, buys at most -2.8% width (Bike), -0.1% (Temperature), and is
        actually negative (+3.1%) on Superconductivity. There is nothing left in the threshold.
      * The width that remains is in how the set's width scales with sigma. Every existing
        calibrator hard-codes that map: CP-VS uses width ∝ sigma, Standard CP uses width ∝ const,
        and MoG-HPD's level set implicitly gives width ∝ sigma*sqrt(2 log(1/(tau sigma sqrt(2pi)))),
        a *fixed* concave map. None of them can tune it, and the right amount of curvature is
        dataset-dependent: Spearman(sigma^2, resid^2) is 0.78 on Synthetic but only 0.19 on
        Temperature, i.e. sigma is a much noisier scale estimate on some datasets than others.

    This class opens exactly those two degrees of freedom, both no-ops at their defaults.

    Surrogate family
    ----------------
    With per-expert means mu_k(x), std devs sigma_k(x) and gate weights w_k(x) (sum_k w_k = 1):

        G                  = exp(mean(log sigma_k))          [frozen on the calibration set]
        tilde_sigma_k(x;c) = G * (sigma_k(x) / G)^c          [shape knob]
        f_c(y|x)           = sum_k w_k(x) N(y; mu_k(x), tilde_sigma_k(x;c)^2)
        sigma_tot(x;c)^2   = sum_k w_k tilde_sigma_k^2 + sum_k w_k (mu_k - yhat)^2

        S_i(c,theta)       = -log f_c(y_i|x_i) - theta * log sigma_tot(x_i;c)   [threshold field]
        C_hat(x)           = { y : log f_c(y|x) >= -t_hat - theta * log sigma_tot(x;c) }

    c < 1 shrinks the model's heteroscedasticity toward homoscedastic, c > 1 amplifies it, c = 1
    is the raw mixture. theta tilts the density threshold by the point's own predictive scale;
    theta = 0 is the single global threshold.

    Every existing calibrator in this package is a point of this grid:

        (c=1, theta=0)      -> MoG_HPD_Calibrator, exactly
        (c=1, theta=1)      -> CP-VS on total variance (exactly when K=1)
        (c=gamma, theta=1)  -> variance-exponent CP-VS (not otherwise implemented here)
        (c=0, theta=0)      -> Standard CP (homoscedastic, single global width)

    so the tuned method cannot be worse than MoG-HPD on the tuning objective, and empirically the
    minimizer is interior on all four datasets.

    (The between-component / epistemic half of the variance decomposition is reshaped by the
    separate SETA-HPD calibrator, SETAHPDCalibrator, which adds a third exponent rho; STA-HPD is the
    rho = 1 special case of it.)

    Why this narrows the sets
    -------------------------
    Coverage pins exactly one scalar (t_hat); the two knobs reallocate that fixed coverage budget
    across x. Differentiating the Lagrangian of "minimize E|C(X)| subject to P(Y in C(X)) >= 1-a",
    the width-optimal rule equalizes the *true* conditional density at the set's boundary,
    p(b(x)|x) = const. MoG-HPD instead enforces f_mix(b(x)|x) = tau_hat, which is optimal only
    where the model density agrees with the truth at the boundary. c corrects the systematic
    sigma-dependent part of that mismatch and theta corrects the residual level. Concretely, when
    sigma carries only partial information about |resid|, dispersion in sigma inflates E[sigma]
    faster than it deflates Quantile_{1-a}(|e|/sigma), so the width-minimizing exponent satisfies
    c* < 1 -- a noise-aware shrinkage of heteroscedasticity no other method here can express.

    Conformal validity
    ------------------
    For any (c, theta) fixed before the calibration labels are seen, S is simply another frozen
    scoring function, so the usual finite-sample split-conformal marginal guarantee holds exactly,
    with no assumption that the mixture is well specified -- only the resulting width depends on
    model quality. Selecting (c, theta) on the same points that set t_hat would break that, so the
    calibration set is split into disjoint tune/calib folds.

    Selection
    ---------
    A single tune/calib split is not usable here: it was measured to *lose* on this repo's data --
    a width-minimizing ratio tuned on one held-out split can win on the tune fold and lose on test
    purely from split noise. A strict 50/50 split halves the data behind t_hat and the resulting
    quantile noise costs more width than the tuning gains -- the gain collapses to roughly zero on
    Temperature and Superconductivity. Two changes fix it:

      * tune_frac = 0.2, not 0.5. The knob grid is 2-dimensional and discrete, so it needs far
        less data than the (1-alpha) quantile does.
      * Repeated splitting with mode-vote and median aggregation: n_repeats independent splits,
        (c*, theta*) is the *majority vote* of the per-split argmins (mode, not mean -- the grid
        is discrete), and t_hat is the median of the per-split calib-fold quantiles at the now
        fixed (c*, theta*). This is the same cross-conformal-style aggregation
        AdaptiveVarianceCalibrator already uses (n_repeats=20, median q_sq).

    Returns
    -------
    predict() returns a **ragged list of length N_test**, whose i-th entry is a float array of
    shape [m_i, 2] holding the sorted disjoint [lower, upper] segments of the prediction set for
    test point i. m_i may be 0 (empty set) or > 1 (multimodal set); it is not a single [N,2] array.
    """

    def __init__(self, alpha=0.1, grid_per_component=200, tail_sigma_mult=6.0,
                 max_tail_expansions=4, merge_eps=1e-6,
                 c_grid=None, theta_grid=None, tune_frac=0.2, n_repeats=10,
                 max_tune_points=500, tune_grid_points=900, tune_sigma_mult=9.0,
                 split_seed=0, single_fold=False, min_split_n=200, verbose=True):
        """
        Parameters beyond MoG_HPD_Calibrator's:

        c_grid           : candidate shape exponents on the per-expert sigmas. Default
                           (0.4, 0.6, 0.8, 1.0, 1.3); 1.0 (the raw mixture) is on it by
                           construction so MoG-HPD is always a candidate.
        theta_grid       : candidate threshold-field exponents. Default
                           (-0.5, 0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0); 0.0 (a single global
                           threshold) is on it, and 1.0 reproduces CP-VS for K=1.
        tune_frac        : fraction of D_cal assigned to D_tune in each split. 0.2 by default --
                           see the class docstring; 0.5 was measured to erase the width gain.
        n_repeats        : number of independent tune/calib splits voted over.
        max_tune_points  : cap on the number of D_tune points used to evaluate the width surface.
        tune_grid_points : resolution of the vectorized fixed-grid measure used *only* inside the
                           tuning loop. The final predict() always uses the parent's exact
                           brentq root-finding; this approximation only affects how precisely
                           (c*, theta*) is estimated, never coverage.
        tune_sigma_mult  : half-width of the tuning grid window, in units of the largest component
                           sigma. Wider than tail_sigma_mult because the fixed grid has no
                           expansion fallback.
        split_seed       : base seed; split b uses RandomState(split_seed + b).
        single_fold      : if True (or if n_cal < min_split_n), tune and take the final quantile on
                           the *same* full calibration set. This is the small-D_cal fallback: it
                           keeps the method usable but the coverage guarantee is no longer exact,
                           because (c*, theta*) is then selected in-sample. A warning is emitted.
        min_split_n      : below this many calibration points the split is not attempted.
        verbose          : emit the structured [STA-HPD] stage logs.
        """
        super().__init__(alpha=alpha, grid_per_component=grid_per_component,
                         tail_sigma_mult=tail_sigma_mult,
                         max_tail_expansions=max_tail_expansions, merge_eps=merge_eps)

        self.c_grid = tuple(c_grid) if c_grid is not None else (0.4, 0.6, 0.8, 1.0, 1.3)
        self.theta_grid = tuple(theta_grid) if theta_grid is not None else \
            (-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
        self.tune_frac = tune_frac
        self.n_repeats = n_repeats
        self.max_tune_points = max_tune_points
        self.tune_grid_points = tune_grid_points
        self.tune_sigma_mult = tune_sigma_mult
        self.split_seed = split_seed
        self.single_fold = single_fold
        self.min_split_n = min_split_n
        self.verbose = verbose

        self.c_ = None                  # chosen shape exponent c*
        self.theta_ = None              # chosen threshold-field exponent theta*
        self.G_ = None                  # geometric-mean sigma anchor, frozen on D_cal
        self.tuning_widths_ = {}        # (c, theta) -> mean tune-fold width, averaged over splits
        self.tuning_widths_std_ = {}    # (c, theta) -> across-split std of that mean
        self.tune_width_at_best_ = None
        self.vote_counts_ = {}          # (c, theta) -> how many splits picked it
        self.n_tune_ = None
        self.n_calib_ = None
        self.n_repeats_ = None
        self.single_fold_used_ = False
        # self.t_hat / self.tau_ are inherited; t_hat is set by fit(), tau_ is only meaningful
        # here as the threshold at sigma_tot = 1 (the field makes the real threshold per-point).

    # ------------------------------------------------------------------ logging

    def _log(self, stage, **fields):
        """One `key=value` line per pipeline stage, so runs can be diffed without re-deriving."""
        if not self.verbose:
            return
        parts = []
        for k, v in fields.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.6g}")
            else:
                parts.append(f"{k}={v}")
        print(f"[STA-HPD] {stage}: " + " ".join(parts))

    @staticmethod
    def _mmm(a):
        """Compact min/median/max summary for a log line."""
        a = np.asarray(a, dtype=float)
        return f"{np.min(a):.4g}/{np.median(a):.4g}/{np.max(a):.4g}"

    # ------------------------------------------------------------------ surrogate family

    def _reshape_sigma(self, sigma, c):
        """
        tilde_sigma_k = G * (sigma_k / G)^c, with G the calibration-set geometric mean frozen by
        fit(). c = 1 is the identity, so the raw mixture is recovered exactly. G must be the same
        object at fit and predict time or the shared threshold is meaningless.
        """
        if self.G_ is None:
            raise ValueError("_reshape_sigma called before fit() froze the sigma anchor G_.")
        if c == 1.0:
            return sigma
        return self.G_ * (sigma / self.G_) ** c

    @staticmethod
    def _log_sigma_tot(mu, sigma_tilde, pi):
        """
        log of the reshaped mixture's total predictive scale,
        sigma_tot^2 = sum_k pi_k tilde_sigma_k^2 + sum_k pi_k (mu_k - yhat)^2
        (law of total variance, matching Exp_Tabular_Regression._collect_predictions).
        mu/sigma_tilde/pi: [N,K] -> [N].
        """
        y_hat = np.sum(pi * mu, axis=1, keepdims=True)
        var = np.sum(pi * sigma_tilde ** 2, axis=1) + np.sum(pi * (mu - y_hat) ** 2, axis=1)
        return 0.5 * np.log(np.clip(var, 1e-300, None))

    def _scores_for(self, trues, mu, sigma_tilde, pi, theta):
        """Nonconformity scores S_i = -log f_c(y_i|x_i) - theta * log sigma_tot(x_i) : [N]."""
        s = -self._log_mix_density(trues, mu, sigma_tilde, pi)
        if theta != 0.0:
            s = s - theta * self._log_sigma_tot(mu, sigma_tilde, pi)
        return s

    def _log_thresholds(self, mu, sigma_tilde, pi, t_hat, theta):
        """
        Per-point log-density threshold log tau_i = -t_hat - theta * log sigma_tot(x_i) : [N].
        theta = 0 collapses to the single global -t_hat of MoG_HPD_Calibrator.
        """
        lt = np.full(mu.shape[0], -float(t_hat))
        if theta != 0.0:
            lt = lt - theta * self._log_sigma_tot(mu, sigma_tilde, pi)
        return lt

    # ------------------------------------------------------------------ tuning-only measure

    def _mean_measure_fast(self, mu, sigma_tilde, pi, log_tau):
        """
        Mean Lebesgue measure of {y : f_c(y|x_i) >= tau_i}, on a vectorized fixed grid.

        Used ONLY to rank (c, theta) candidates inside the tuning loop: the exact
        _segments_for_point path costs a grid scan plus a brentq solve per crossing, and the
        surface here is |c_grid| * |theta_grid| * n_repeats evaluations. The final predict()
        always uses the parent's exact root-finding, so this approximation can only perturb which
        grid cell wins -- never the conformal threshold and never coverage.

        mu/sigma_tilde/pi: [N,K]; log_tau: [N] -> scalar mean measure.
        """
        lo = np.min(mu - self.tune_sigma_mult * sigma_tilde, axis=1)
        hi = np.max(mu + self.tune_sigma_mult * sigma_tilde, axis=1)
        t = np.linspace(0.0, 1.0, self.tune_grid_points)[None, :]
        grid = lo[:, None] + (hi - lo)[:, None] * t                      # [N,M]
        dy = (hi - lo) / (self.tune_grid_points - 1)                     # [N]

        # log f_c on the grid, same mixture form as the parent's _log_mix_density_1d but batched
        # over points instead of looping.
        z = (grid[:, :, None] - mu[:, None, :]) / sigma_tilde[:, None, :]
        log_comp = (-0.5 * np.log(2 * np.pi) - np.log(sigma_tilde)[:, None, :] - 0.5 * z ** 2
                    + np.log(np.clip(pi, 1e-12, None))[:, None, :])
        m = np.max(log_comp, axis=2)
        log_f = m + np.log(np.sum(np.exp(log_comp - m[:, :, None]), axis=2))  # [N,M]

        return float(np.mean(np.sum(log_f >= log_tau[:, None], axis=1) * dy))

    # ------------------------------------------------------------------ fit / predict

    def fit(self, cal_trues, cal_mu, cal_sigma, cal_pi):
        """
        Same signature as MoG_HPD_Calibrator.fit, so this class is a drop-in at the call site.
        cal_trues: [N] (or [N,1]); cal_mu / cal_sigma / cal_pi: [N,K].
        """
        cal_trues = self._to_numpy(cal_trues).squeeze()
        cal_mu = self._to_numpy(cal_mu)
        cal_sigma = self._to_numpy(cal_sigma)
        cal_pi = self._to_numpy(cal_pi)

        n_total = len(cal_trues)
        K = cal_mu.shape[1]

        # Freeze the sigma anchor on the FULL calibration set, before any fold is drawn: it is a
        # property of the scoring function, not of a particular split.
        self.G_ = float(np.exp(np.mean(np.log(np.clip(cal_sigma, 1e-300, None)))))
        self._log("sigma_anchor", n=n_total, K=K, G=self.G_,
                  sigma_min_med_max=self._mmm(cal_sigma))

        self.single_fold_used_ = bool(self.single_fold or n_total < self.min_split_n)
        if self.single_fold_used_:
            if not self.single_fold:
                print(f"[STA-HPD] WARNING: only {n_total} calibration points "
                      f"(< min_split_n={self.min_split_n}); falling back to a single fold. "
                      f"(c*, theta*) is then chosen in-sample, so the coverage guarantee is no "
                      f"longer exact.")
            else:
                print("[STA-HPD] WARNING: single_fold=True -- (c*, theta*) is chosen on the same "
                      "data used for the final quantile, so the coverage guarantee is no longer "
                      "exact.")
            splits = [(np.arange(n_total), np.arange(n_total))]
            self.n_repeats_ = 1
        else:
            n_tune = int(round(self.tune_frac * n_total))
            splits = []
            for b in range(self.n_repeats):
                perm = np.random.RandomState(self.split_seed + b).permutation(n_total)
                splits.append((perm[:n_tune], perm[n_tune:]))
            self.n_tune_ = n_tune
            self.n_calib_ = n_total - n_tune
            self.n_repeats_ = self.n_repeats
        self._log("splits", n_repeats=self.n_repeats_, tune_frac=self.tune_frac,
                  n_tune=self.n_tune_, n_calib=self.n_calib_,
                  single_fold=self.single_fold_used_)

        # Reshaped sigmas per candidate c, computed once and reused across theta and splits.
        sigma_by_c = {c: self._reshape_sigma(cal_sigma, c) for c in self.c_grid}
        for c in self.c_grid:
            st = sigma_by_c[c]
            lst = self._log_sigma_tot(cal_mu, st, cal_pi)
            self._log("sigma_reshape", c=c, sigma_tilde_min_med_max=self._mmm(st),
                      log_sigma_tot_mean=float(np.mean(lst)),
                      log_sigma_tot_std=float(np.std(lst)))

        # --- Step 1: rank the (c, theta) surface on each split's tune fold, then majority-vote.
        keys = [(c, th) for c in self.c_grid for th in self.theta_grid]
        width_curves = np.empty((len(splits), len(keys)))
        votes = []

        for b, (tune_idx, _) in enumerate(splits):
            # Subsample only the *width evaluation*: t_hat(c,theta) still uses the whole tune
            # fold, so the threshold being measured against is the one that fold actually implies.
            if self.max_tune_points is not None and len(tune_idx) > self.max_tune_points:
                sub_idx = np.random.RandomState(self.split_seed + 1000 + b).choice(
                    tune_idx, size=self.max_tune_points, replace=False)
            else:
                sub_idx = tune_idx

            for j, (c, th) in enumerate(keys):
                st = sigma_by_c[c]
                scores = self._scores_for(cal_trues[tune_idx], cal_mu[tune_idx],
                                          st[tune_idx], cal_pi[tune_idx], th)
                t_hat = float(np.quantile(scores, self._q_level(len(scores)), method='higher'))
                log_tau = self._log_thresholds(cal_mu[sub_idx], st[sub_idx], cal_pi[sub_idx],
                                               t_hat, th)
                width_curves[b, j] = self._mean_measure_fast(cal_mu[sub_idx], st[sub_idx],
                                                             cal_pi[sub_idx], log_tau)
            votes.append(keys[int(np.argmin(width_curves[b]))])

        avg = width_curves.mean(axis=0)
        std = width_curves.std(axis=0)
        self.tuning_widths_ = {k: float(w) for k, w in zip(keys, avg)}
        self.tuning_widths_std_ = {k: float(s) for k, s in zip(keys, std)}
        self.vote_counts_ = dict(Counter(votes))

        # Majority vote across splits. Ties (and the single-fold case) fall back to the smallest
        # average tune width among the tied cells, then to the cell closest to MoG-HPD (1, 0) --
        # so a flat surface keeps the established default rather than an arbitrary cell.
        top = max(self.vote_counts_.values())
        tied = [k for k, v in self.vote_counts_.items() if v == top]
        best = min(tied, key=lambda k: (self.tuning_widths_[k],
                                        abs(k[0] - 1.0) + abs(k[1])))
        self.c_, self.theta_ = float(best[0]), float(best[1])
        self.tune_width_at_best_ = float(self.tuning_widths_[best])

        on_c_edge = self.c_ in (self.c_grid[0], self.c_grid[-1])
        on_th_edge = self.theta_ in (self.theta_grid[0], self.theta_grid[-1])
        self._log("tuning", n_cells=len(keys),
                  best_c=self.c_, best_theta=self.theta_,
                  votes=f"{self.vote_counts_[best]}/{len(splits)}",
                  tune_width=self.tune_width_at_best_,
                  tune_width_std=self.tuning_widths_std_[best],
                  width_range=f"[{avg.min():.4f},{avg.max():.4f}]",
                  solution="BOUNDARY" if (on_c_edge or on_th_edge) else "interior",
                  vote_spread=sorted(self.vote_counts_.items(), key=lambda kv: -kv[1])[:3])

        # --- Step 2: final conformal threshold at the now-fixed (c*, theta*), on the calib folds.
        st_best = sigma_by_c[best[0]]
        t_hats = []
        for _, calib_idx in splits:
            scores = self._scores_for(cal_trues[calib_idx], cal_mu[calib_idx],
                                      st_best[calib_idx], cal_pi[calib_idx], self.theta_)
            t_hats.append(float(np.quantile(scores, self._q_level(len(scores)), method='higher')))

        final_scores = self._scores_for(cal_trues, cal_mu, st_best, cal_pi, self.theta_)
        self._log("scores", c=self.c_, theta=self.theta_,
                  score_min_med_max=self._mmm(final_scores),
                  n_nonfinite=int(np.sum(~np.isfinite(final_scores))),
                  q_level=float(self._q_level(self.n_calib_ or n_total)),
                  n=n_total)

        self.t_hat = float(np.median(t_hats))
        self.tau_ = float(np.exp(-self.t_hat))
        self._log("quantile", t_hat_median=self.t_hat, tau_at_unit_sigma=self.tau_,
                  t_hat_per_split=np.round(t_hats, 4).tolist())

    def predict(self, test_mu, test_sigma, test_pi):
        """
        Prediction sets {y : f_{c*}(y|x) >= tau_i}, tau_i = exp(-t_hat - theta* log sigma_tot(x_i)).
        Returns a ragged list of [m_i, 2] arrays (see the class docstring); m_i = 0 means an empty
        prediction set. Unlike the parent, the threshold varies per point, so this cannot delegate
        to super().predict() -- it drives _segments_for_point directly, keeping the exact brentq
        root refinement.
        """
        if self.c_ is None or self.t_hat is None:
            raise ValueError("Calibrator must be fitted before calling predict.")

        test_mu = self._to_numpy(test_mu)
        test_sigma = self._to_numpy(test_sigma)
        test_pi = self._to_numpy(test_pi)

        sigma_tilde = self._reshape_sigma(test_sigma, self.c_)
        log_tau = self._log_thresholds(test_mu, sigma_tilde, test_pi, self.t_hat, self.theta_)
        taus = np.exp(log_tau)

        intervals = []
        for i in range(test_mu.shape[0]):
            intervals.append(self._segments_for_point(test_mu[i], sigma_tilde[i], test_pi[i],
                                                      taus[i]))

        widths = np.array([float(np.sum(s[:, 1] - s[:, 0])) if len(s) else 0.0 for s in intervals])
        n_seg = np.array([len(s) for s in intervals])
        self._log("predict_widths", n=len(intervals), c=self.c_, theta=self.theta_,
                  width_mean=float(widths.mean()), width_median=float(np.median(widths)),
                  width_p90=float(np.percentile(widths, 90)),
                  avg_segments=float(n_seg.mean()),
                  frac_multi=float(np.mean(n_seg > 1)), frac_empty=float(np.mean(n_seg == 0)),
                  n_zero_width=int(np.sum(widths <= 0.0)),
                  n_nonfinite=int(np.sum(~np.isfinite(widths))),
                  tau_min_med_max=self._mmm(taus))
        return intervals
