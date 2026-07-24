import numpy as np
from collections import Counter

from calibration.tabular_regression.MoG_HPD_Calibrator import MoG_HPD_Calibrator


class SETAHPDCalibrator(MoG_HPD_Calibrator):
    """
    SETA-HPD -- Shape-Epistemic-Threshold Adaptive HPD conformal calibration for
    Mixture-of-Gaussians models.

    SETA-HPD is the MoG-structure-aware generalization of STA-HPD (STAHPDCalibrator). STA-HPD reshapes
    only the WITHIN-component (aleatoric) half of the MoG variance decomposition (exponent c) plus a
    threshold field (theta). SETA-HPD additionally reshapes the BETWEEN-component (epistemic) half --
    the expert disagreement sum_k w_k (mu_k - yhat)^2 -- with its own exponent rho, so both terms of
    the law of total variance are tuned. Setting rho = 1 (mog_decompose=False, or rho_grid=(1.0,))
    reduces SETA-HPD exactly to STA-HPD; the two live in separate files so either can be run on its
    own (STA-HPD via calibrate_sta_hpd / --use_reg_sta_hpd, SETA-HPD via calibrate_seta_hpd /
    --use_reg_seta_hpd).

    Motivation
    ----------
    MoG_HPD_Calibrator conformalizes S = -log f_mix(y|x) and returns the global super-level set
    {y : f_mix(y|x) >= tau_hat}. By the Neyman-Pearson argument in docs/calibration_methods.tex
    that is the minimum-measure set achieving a given coverage *under the density it is handed*, but
    only when the model density equals the truth. STA-HPD corrects the WITHIN-component scale (c) and
    the threshold level (theta). What it leaves untouched is the BETWEEN-component (epistemic) spread:
    an MoE trained with load-balancing routinely OVER-separates its experts, opening low-density
    valleys between well-separated modes that inflate the HPD set's width (and fragment it into
    disjoint intervals). SETA-HPD opens that third degree of freedom.

    Surrogate family
    ----------------
    With per-expert means mu_k(x), std devs sigma_k(x), gate weights w_k(x) (sum_k w_k = 1) and
    the aggregate prediction yhat(x) = sum_k w_k(x) mu_k(x), SETA-HPD reshapes *both* terms of the
    Mixture-of-Gaussians law-of-total-variance decomposition
    Var(Y|x) = sum_k w_k sigma_k^2 (within-component / aleatoric) + sum_k w_k (mu_k - yhat)^2
    (between-component / epistemic), each with its own exponent, then thresholds the resulting
    surrogate density:

        G                    = exp(mean(log sigma_k))                 [frozen on the calibration set]
        tilde_sigma_k(x;c)   = G * (sigma_k(x) / G)^c                 [WITHIN-component shape knob]
        tilde_mu_k(x;rho)    = yhat(x) + sqrt(rho) * (mu_k(x) - yhat) [BETWEEN-component shape knob]
        f_{c,rho}(y|x)       = sum_k w_k(x) N(y; tilde_mu_k(x;rho), tilde_sigma_k(x;c)^2)
        sigma_tot(x;c,rho)^2 = sum_k w_k tilde_sigma_k^2 + rho * sum_k w_k (mu_k - yhat)^2

        S_i(c,rho,theta)     = -log f_{c,rho}(y_i|x_i) - theta * log sigma_tot(x_i;c,rho)  [field]
        C_hat(x)             = { y : log f_{c,rho}(y|x) >= -t_hat - theta * log sigma_tot(x;c,rho) }

    The mean shift tilde_mu_k = yhat + sqrt(rho)(mu_k - yhat) preserves the weighted mean exactly
    (sum_k w_k tilde_mu_k = yhat) and multiplies the between-component (epistemic) variance by
    exactly rho, leaving the aleatoric part to the c knob -- the two exponents act on disjoint terms
    of the variance decomposition. c < 1 shrinks per-expert heteroscedasticity toward homoscedastic,
    c > 1 amplifies it, c = 1 is the raw within-component scale. rho < 1 compresses experts toward
    yhat (filling the density valleys that over-separated experts open up, favouring a single
    interval), rho > 1 amplifies expert disagreement, rho = 1 is the raw between-component scale
    (i.e. plain STA-HPD). theta tilts the density threshold by the point's own predictive scale;
    theta = 0 is a single global threshold. rho has effect only when K > 1; for a single component the
    between-component term is identically zero and SETA-HPD coincides with STA-HPD.

    Boundary recovery
    -----------------
    Every existing calibrator in this package is a point of this grid:

        (c=1, rho=1, theta=0)   -> MoG_HPD_Calibrator, exactly
        (c=1, rho=*, theta=0)   -> epistemic-scale-only HPD (rho rescales the between-component term)
        (c=*, rho=1, theta=*)   -> STA-HPD (STAHPDCalibrator)
        (c=1, rho=1, theta=1)   -> CP-VS on total variance (exactly when K=1)
        (c=0, rho=1, theta=0)   -> Standard CP (homoscedastic, single global width)

    so the tuned method cannot be worse than MoG-HPD or STA-HPD on the tuning objective, since both
    are members of its search grid (1.0 is always kept on rho_grid).

    Conformal validity
    ------------------
    For any (c, rho, theta) fixed before the calibration labels are seen, S is simply another frozen
    scoring function, so the usual finite-sample split-conformal marginal guarantee holds exactly,
    with no assumption that the mixture is well specified -- only the resulting width depends on
    model quality. Selecting (c, rho, theta) on the same points that set t_hat would break that, so
    the calibration set is split into disjoint tune/calib folds.

    Selection
    ---------
    n_repeats independent tune/calib splits; (c*, rho*, theta*) is the *majority vote* of the
    per-split argmins (mode, appropriate for a discrete grid), and t_hat is the median of the
    per-split calib-fold quantiles at the now fixed (c*, rho*, theta*) -- the same cross-conformal
    style stabilization STA-HPD and AdaptiveVarianceCalibrator use.

    Returns
    -------
    predict() returns a **ragged list of length N_test**, whose i-th entry is a float array of
    shape [m_i, 2] holding the sorted disjoint [lower, upper] segments of the prediction set for
    test point i. m_i may be 0 (empty set) or > 1 (multimodal set); it is not a single [N,2] array.
    """

    def __init__(self, alpha=0.1, grid_per_component=200, tail_sigma_mult=6.0,
                 max_tail_expansions=4, merge_eps=1e-6,
                 c_grid=None, theta_grid=None, rho_grid=None, mog_decompose=True,
                 tune_frac=0.2, n_repeats=10,
                 max_tune_points=500, tune_grid_points=900, tune_sigma_mult=9.0,
                 split_seed=0, single_fold=False, min_split_n=200, verbose=True):
        """
        Parameters beyond MoG_HPD_Calibrator's:

        c_grid           : candidate WITHIN-component (aleatoric) shape exponents on the per-expert
                           sigmas. Default (0.4, 0.6, 0.8, 1.0, 1.3); 1.0 (the raw mixture) is on it
                           by construction so MoG-HPD is always a candidate.
        theta_grid       : candidate threshold-field exponents. Default
                           (-0.5, 0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0); 0.0 (a single global
                           threshold) is on it, and 1.0 reproduces CP-VS for K=1.
        rho_grid         : candidate BETWEEN-component (epistemic) scale exponents -- the knob that
                           distinguishes SETA-HPD from STA-HPD. Default None, which resolves to
                           (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0) when mog_decompose=True (the default
                           here) or to (1.0,) when mog_decompose=False (which makes SETA-HPD identical
                           to STA-HPD). 1.0 (the raw between-component scale / plain STA-HPD) is always
                           kept on the grid. Passing an explicit rho_grid overrides mog_decompose.
        mog_decompose    : convenience switch for rho_grid. True (default for SETA-HPD) => the default
                           epistemic grid above. False => rho fixed at 1 (== STA-HPD). Ignored when
                           rho_grid is passed explicitly.
        tune_frac        : fraction of D_cal assigned to D_tune in each split. 0.2 by default.
        n_repeats        : number of independent tune/calib splits voted over.
        max_tune_points  : cap on the number of D_tune points used to evaluate the width surface.
        tune_grid_points : resolution of the vectorized fixed-grid measure used *only* inside the
                           tuning loop. The final predict() always uses the parent's exact
                           brentq root-finding; this approximation only affects how precisely
                           (c*, rho*, theta*) is estimated, never coverage.
        tune_sigma_mult  : half-width of the tuning grid window, in units of the largest component
                           sigma. Wider than tail_sigma_mult because the fixed grid has no
                           expansion fallback.
        split_seed       : base seed; split b uses RandomState(split_seed + b).
        single_fold      : if True (or if n_cal < min_split_n), tune and take the final quantile on
                           the *same* full calibration set. This is the small-D_cal fallback: it
                           keeps the method usable but the coverage guarantee is no longer exact,
                           because (c*, rho*, theta*) is then selected in-sample. A warning is emitted.
        min_split_n      : below this many calibration points the split is not attempted.
        verbose          : emit the structured [SETA-HPD] stage logs.
        """
        super().__init__(alpha=alpha, grid_per_component=grid_per_component,
                         tail_sigma_mult=tail_sigma_mult,
                         max_tail_expansions=max_tail_expansions, merge_eps=merge_eps)

        self.c_grid = tuple(c_grid) if c_grid is not None else (0.4, 0.6, 0.8, 1.0, 1.3)
        self.theta_grid = tuple(theta_grid) if theta_grid is not None else \
            (-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
        if rho_grid is not None:
            self.rho_grid = tuple(rho_grid)
        elif mog_decompose:
            self.rho_grid = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
        else:
            self.rho_grid = (1.0,)
        # 1.0 must be on the grid so the STA-HPD family (rho off) is always reachable.
        if 1.0 not in self.rho_grid:
            self.rho_grid = tuple(sorted(self.rho_grid + (1.0,)))
        # The epistemic mechanism is "active" iff rho is allowed to move off 1.
        self.mog_decompose = (self.rho_grid != (1.0,))
        self.tune_frac = tune_frac
        self.n_repeats = n_repeats
        self.max_tune_points = max_tune_points
        self.tune_grid_points = tune_grid_points
        self.tune_sigma_mult = tune_sigma_mult
        self.split_seed = split_seed
        self.single_fold = single_fold
        self.min_split_n = min_split_n
        self.verbose = verbose

        self.c_ = None                  # chosen within-component (aleatoric) shape exponent c*
        self.rho_ = None                # chosen between-component (epistemic) scale exponent rho*
        self.theta_ = None              # chosen threshold-field exponent theta*
        self.G_ = None                  # geometric-mean sigma anchor, frozen on D_cal
        self.tuning_widths_ = {}        # (c, rho, theta) -> mean tune-fold width, averaged over splits
        self.tuning_widths_std_ = {}    # (c, rho, theta) -> across-split std of that mean
        self.tune_width_at_best_ = None
        self.vote_counts_ = {}          # (c, rho, theta) -> how many splits picked it
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
        print(f"[SETA-HPD] {stage}: " + " ".join(parts))

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
    def _shift_means(mu, pi, rho):
        """
        Reshape the BETWEEN-component (epistemic) spread: tilde_mu_k = yhat + sqrt(rho)(mu_k - yhat)
        with yhat = sum_k pi_k mu_k. This preserves the weighted mean exactly and scales the
        between-component variance sum_k pi_k (mu_k - yhat)^2 by exactly rho. mu/pi: [N,K] -> [N,K].
        rho = 1 is the identity (returns mu unchanged), so the epistemic mechanism is a no-op; rho = 0
        collapses every expert onto yhat. Downstream helpers (_log_sigma_tot, _scores_for,
        _grid_log_density) are simply handed these shifted means, so the epistemic term of
        sigma_tot picks up the factor rho automatically (yhat is preserved by the shift).
        """
        if rho == 1.0:
            return mu
        y_hat = np.sum(pi * mu, axis=1, keepdims=True)
        return y_hat + np.sqrt(rho) * (mu - y_hat)

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
        """Nonconformity scores S_i = -log f_{c,rho}(y_i|x_i) - theta * log sigma_tot(x_i) : [N]."""
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

    def _grid_log_density(self, mu, sigma_tilde, pi):
        """
        Vectorized log f_{c,rho} on a per-point fixed grid, plus the grid step. This is the part of
        the tuning measure that depends only on the density (c, rho), NOT on the threshold field
        theta. Factoring it out lets the tuning loop compute it once per (c, rho) and then sweep the
        whole theta grid with only cheap threshold-count operations (an ~|theta_grid|x speedup, since
        the density -- the expensive [N,M,K] mixture evaluation -- no longer repeats per theta).

        mu/sigma_tilde/pi: [N,K] -> (log_f: [N,M], dy: [N]).
        """
        lo = np.min(mu - self.tune_sigma_mult * sigma_tilde, axis=1)
        hi = np.max(mu + self.tune_sigma_mult * sigma_tilde, axis=1)
        t = np.linspace(0.0, 1.0, self.tune_grid_points)[None, :]
        grid = lo[:, None] + (hi - lo)[:, None] * t                      # [N,M]
        dy = (hi - lo) / (self.tune_grid_points - 1)                     # [N]

        # log f on the grid, same mixture form as the parent's _log_mix_density_1d but batched
        # over points instead of looping.
        z = (grid[:, :, None] - mu[:, None, :]) / sigma_tilde[:, None, :]
        log_comp = (-0.5 * np.log(2 * np.pi) - np.log(sigma_tilde)[:, None, :] - 0.5 * z ** 2
                    + np.log(np.clip(pi, 1e-12, None))[:, None, :])
        m = np.max(log_comp, axis=2)
        log_f = m + np.log(np.sum(np.exp(log_comp - m[:, :, None]), axis=2))  # [N,M]
        return log_f, dy

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
                print(f"[SETA-HPD] WARNING: only {n_total} calibration points "
                      f"(< min_split_n={self.min_split_n}); falling back to a single fold. "
                      f"(c*, rho*, theta*) is then chosen in-sample, so the coverage guarantee is no "
                      f"longer exact.")
            else:
                print("[SETA-HPD] WARNING: single_fold=True -- (c*, rho*, theta*) is chosen on the "
                      "same data used for the final quantile, so the coverage guarantee is no longer "
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

        # Reshaped sigmas per candidate c (WITHIN-component) and mean-shifted means per candidate rho
        # (BETWEEN-component), each computed once on the full calibration set and reused across the
        # other two axes and all splits. rho = 1 leaves cal_mu untouched (see _shift_means), so with
        # rho_grid = (1.0,) (mog_decompose=False) the search is the STA-HPD (c, theta) grid.
        sigma_by_c = {c: self._reshape_sigma(cal_sigma, c) for c in self.c_grid}
        mu_by_rho = {rho: self._shift_means(cal_mu, cal_pi, rho) for rho in self.rho_grid}
        for c in self.c_grid:
            st = sigma_by_c[c]
            lst = self._log_sigma_tot(cal_mu, st, cal_pi)
            self._log("sigma_reshape", c=c, sigma_tilde_min_med_max=self._mmm(st),
                      log_sigma_tot_mean=float(np.mean(lst)),
                      log_sigma_tot_std=float(np.std(lst)))

        # --- Step 1: rank the (c, rho, theta) surface on each split's tune fold, then majority-vote.
        # The density f_{c,rho} does not depend on theta, so for each (c, rho) the grid density and
        # the two log-sigma_tot vectors are computed once and the whole theta grid is swept with only
        # cheap quantile + threshold-count ops (roughly a |theta_grid|x speedup over recomputing the
        # [N,M,K] mixture per theta).
        keys = [(c, rho, th) for c in self.c_grid for rho in self.rho_grid
                for th in self.theta_grid]
        key_col = {k: j for j, k in enumerate(keys)}
        width_curves = np.empty((len(splits), len(keys)))
        votes = []

        for b, (tune_idx, _) in enumerate(splits):
            # Subsample only the *width evaluation*: t_hat(c,rho,theta) still uses the whole tune
            # fold, so the threshold being measured against is the one that fold actually implies.
            if self.max_tune_points is not None and len(tune_idx) > self.max_tune_points:
                sub_idx = np.random.RandomState(self.split_seed + 1000 + b).choice(
                    tune_idx, size=self.max_tune_points, replace=False)
            else:
                sub_idx = tune_idx
            q_lvl = self._q_level(len(tune_idx))

            for c in self.c_grid:
                st = sigma_by_c[c]
                for rho in self.rho_grid:
                    mu_r = mu_by_rho[rho]  # epistemic-reshaped means; sigma_tot picks up rho via these
                    # (c, rho)-dependent, theta-independent pieces, computed once:
                    neg_logdens_tune = -self._log_mix_density(cal_trues[tune_idx], mu_r[tune_idx],
                                                              st[tune_idx], cal_pi[tune_idx])  # [Ntune]
                    lst_tune = self._log_sigma_tot(mu_r[tune_idx], st[tune_idx], cal_pi[tune_idx])
                    log_f_sub, dy_sub = self._grid_log_density(mu_r[sub_idx], st[sub_idx],
                                                               cal_pi[sub_idx])                # [Nsub,M]
                    lst_sub = self._log_sigma_tot(mu_r[sub_idx], st[sub_idx], cal_pi[sub_idx])  # [Nsub]
                    for th in self.theta_grid:
                        scores = neg_logdens_tune - th * lst_tune
                        t_hat = float(np.quantile(scores, q_lvl, method='higher'))
                        log_tau = -t_hat - th * lst_sub                                        # [Nsub]
                        width = float(np.mean(np.sum(log_f_sub >= log_tau[:, None], axis=1) * dy_sub))
                        width_curves[b, key_col[(c, rho, th)]] = width
            votes.append(keys[int(np.argmin(width_curves[b]))])

        avg = width_curves.mean(axis=0)
        std = width_curves.std(axis=0)
        self.tuning_widths_ = {k: float(w) for k, w in zip(keys, avg)}
        self.tuning_widths_std_ = {k: float(s) for k, s in zip(keys, std)}
        self.vote_counts_ = dict(Counter(votes))

        # Majority vote across splits. Ties (and the single-fold case) fall back to the smallest
        # average tune width among the tied cells, then to the cell closest to MoG-HPD (1, 1, 0) --
        # so a flat surface keeps the established default rather than an arbitrary cell.
        top = max(self.vote_counts_.values())
        tied = [k for k, v in self.vote_counts_.items() if v == top]
        best = min(tied, key=lambda k: (self.tuning_widths_[k],
                                        abs(k[0] - 1.0) + abs(k[1] - 1.0) + abs(k[2])))
        self.c_, self.rho_, self.theta_ = float(best[0]), float(best[1]), float(best[2])
        self.tune_width_at_best_ = float(self.tuning_widths_[best])

        on_c_edge = self.c_ in (self.c_grid[0], self.c_grid[-1])
        on_rho_edge = self.mog_decompose and self.rho_ in (self.rho_grid[0], self.rho_grid[-1])
        on_th_edge = self.theta_ in (self.theta_grid[0], self.theta_grid[-1])
        self._log("tuning", n_cells=len(keys),
                  best_c=self.c_, best_rho=self.rho_, best_theta=self.theta_,
                  mog_decompose=self.mog_decompose,
                  votes=f"{self.vote_counts_[best]}/{len(splits)}",
                  tune_width=self.tune_width_at_best_,
                  tune_width_std=self.tuning_widths_std_[best],
                  width_range=f"[{avg.min():.4f},{avg.max():.4f}]",
                  solution="BOUNDARY" if (on_c_edge or on_rho_edge or on_th_edge) else "interior",
                  vote_spread=sorted(self.vote_counts_.items(), key=lambda kv: -kv[1])[:3])

        # --- Step 2: final conformal threshold at the now-fixed (c*, rho*, theta*), on calib folds.
        st_best = sigma_by_c[best[0]]
        mu_best = mu_by_rho[best[1]]
        t_hats = []
        for _, calib_idx in splits:
            scores = self._scores_for(cal_trues[calib_idx], mu_best[calib_idx],
                                      st_best[calib_idx], cal_pi[calib_idx], self.theta_)
            t_hats.append(float(np.quantile(scores, self._q_level(len(scores)), method='higher')))

        final_scores = self._scores_for(cal_trues, mu_best, st_best, cal_pi, self.theta_)
        self._log("scores", c=self.c_, rho=self.rho_, theta=self.theta_,
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
        Prediction sets {y : f_{c*,rho*}(y|x) >= tau_i},
        tau_i = exp(-t_hat - theta* log sigma_tot(x_i; c*, rho*)). The between-component reshaping
        rho* enters through the epistemic-shifted means tilde_mu_k = yhat + sqrt(rho*)(mu_k - yhat).
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

        mu_shift = self._shift_means(test_mu, test_pi, self.rho_)
        sigma_tilde = self._reshape_sigma(test_sigma, self.c_)
        log_tau = self._log_thresholds(mu_shift, sigma_tilde, test_pi, self.t_hat, self.theta_)
        taus = np.exp(log_tau)

        intervals = []
        for i in range(test_mu.shape[0]):
            intervals.append(self._segments_for_point(mu_shift[i], sigma_tilde[i], test_pi[i],
                                                      taus[i]))

        widths = np.array([float(np.sum(s[:, 1] - s[:, 0])) if len(s) else 0.0 for s in intervals])
        n_seg = np.array([len(s) for s in intervals])
        self._log("predict_widths", n=len(intervals), c=self.c_, rho=self.rho_, theta=self.theta_,
                  width_mean=float(widths.mean()), width_median=float(np.median(widths)),
                  width_p90=float(np.percentile(widths, 90)),
                  avg_segments=float(n_seg.mean()),
                  frac_multi=float(np.mean(n_seg > 1)), frac_empty=float(np.mean(n_seg == 0)),
                  n_zero_width=int(np.sum(widths <= 0.0)),
                  n_nonfinite=int(np.sum(~np.isfinite(widths))),
                  tau_min_med_max=self._mmm(taus))
        return intervals
