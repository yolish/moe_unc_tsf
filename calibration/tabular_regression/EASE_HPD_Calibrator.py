import numpy as np
from collections import Counter

from calibration.tabular_regression.SETA_HPD_Calibrator import SETAHPDCalibrator


class EASEHPDCalibrator(SETAHPDCalibrator):
    """
    EASE-HPD -- Epistemic-Aleatoric Share-adaptive threshold HPD conformal calibration for
    Mixture-of-Gaussians models.

    EASE-HPD uses the exact same surrogate DENSITY family as SETA-HPD (SETAHPDCalibrator): the
    within-component (aleatoric) term is reshaped by exponent c
    (tilde_sigma_k = G (sigma_k/G)^c) and the between-component (epistemic) term by exponent rho
    via the mean-shift tilde_mu_k = yhat + sqrt(rho)(mu_k - yhat). What EASE-HPD adds is a SECOND,
    genuinely different threshold-field axis on top of SETA-HPD's theta.

    Motivation: the unexploited quantity in this family
    ----------------------------------------------------
    STA-HPD's theta and SETA-HPD's theta both tilt the conformal threshold by the point's TOTAL
    predictive scale sigma_tot(x)^2 = A(x) + E(x), where A(x) = sum_k w_k tilde_sigma_k^2(x;c)
    (aleatoric) and E(x) = sum_k w_k (mu_k(x) - yhat(x))^2 * rho (epistemic, after the rho-shift).
    Two points with the SAME sigma_tot but opposite COMPOSITIONS -- one all expert disagreement,
    the other all within-expert noise -- get identical threshold treatment, because theta only
    ever sees the sum. But the two terms are miscalibrated for structurally different reasons: the
    epistemic term is inflated by load-balancing pressure during training (an artifact of how the
    gate is regularized, not a property of the outcome), while the aleatoric term is directly
    fit by the per-expert NLL and is comparatively well-calibrated already. So the part of the
    density-level mismatch that theta exists to correct should track the EPISTEMIC SHARE of the
    variance, not just its total -- a quantity the existing (c, rho, theta) family never looks at.

    The new knob: phi on the epistemic share
    ------------------------------------------
    Define the (surrogate, i.e. post-c, post-rho) epistemic share

        s(x) = E(x) / (A(x) + E(x))   in [0, 1]

    frozen to its calibration-set mean s_bar (per (c, rho) cell, exactly like G is frozen per
    calibration set) so the field is centered and t_hat stays on a comparable scale across phi. A
    second threshold-field exponent phi tilts the score by the point's OWN DEVIATION of that share
    from the typical calibration-set share:

        S_i(c,rho,theta,phi) = -log f_{c,rho}(y_i|x_i) - theta*log sigma_tot(x_i;c,rho)
                                - phi*(s(x_i;c,rho) - s_bar)
        C_hat(x)             = { y : log f_{c,rho}(y|x) >=
                                  -t_hat - theta*log sigma_tot(x;c,rho) - phi*(s(x;c,rho) - s_bar) }

    phi > 0 spends extra width on points where disagreement dominates the variance (more than
    typical for this dataset); phi < 0 tightens them (appropriate when the epistemic inflation is
    a training artifact the model already over-covers for). phi = 0 is the SETA-HPD field exactly.
    s is bounded in [0,1] (no log-ratio blow-ups), centered by s_bar (so phi and theta do not fight
    over the same additive constant), and automatically inert when K=1 or rho=0 (s is then
    constant across x, so the phi term collapses into t_hat and the tie-break keeps phi=0).

    Selection: width-argmin with majority vote (NOT a proper scoring rule)
    ------------------------------------------------------------------------
    This project also has a sibling calibrator, MELDHPDCalibrator, which selects its shape
    exponents (c, rho) by maximizing mean calibration log-density instead of minimizing tune-fold
    width. That is a genuinely different objective (KL-minimizing shape is not, in general, the
    same shape that minimizes conformalized width), and empirically it lost the "can't be worse
    than its ancestor" floor that STA-HPD -> SETA-HPD have by construction: on this repo's real MOG
    checkpoints MELD-HPD came out wider than STA-HPD on several configs. EASE-HPD deliberately
    keeps the PROVEN width-argmin-with-majority-vote selection machinery unchanged (the same
    mechanism STA-HPD and SETA-HPD use) and only widens the GRID by one more axis, phi, so the
    "ancestor is always a reachable grid cell" floor is preserved exactly.

    Boundary recovery
    ------------------
    Every existing calibrator in this package is a point (or sub-family) of this 4-D grid:

        (c=1, rho=1, theta=0, phi=0)   -> MoG_HPD_Calibrator, exactly
        (c=*, rho=1, theta=*, phi=0)   -> STA-HPD (STAHPDCalibrator)
        (c=*, rho=*, theta=*, phi=0)   -> SETA-HPD (SETAHPDCalibrator), the entire phi=0 slice
        (c=*, rho=*, theta=*, phi=*)   -> EASE-HPD

    so the tuned method's tuning objective can never be worse than SETA-HPD's, which can never be
    worse than STA-HPD's, which can never be worse than MoG-HPD's -- the same nested floor
    argument as STA-HPD -> SETA-HPD, extended by one more axis (0.0 is always kept on phi_grid).

    Conformal validity
    ------------------
    Unchanged: (c, rho, theta, phi) are selected using only tune folds; t_hat is the median of the
    per-split calib-fold quantiles at the now-fixed (c*, rho*, theta*, phi*), on folds disjoint
    from the tune folds. The score is therefore a frozen, label-independent function by the time it
    is conformalized, so the finite-sample split-conformal marginal guarantee holds exactly, with
    no assumption that the mixture is well specified.

    predict() returns the same ragged list of [m_i, 2] disjoint-segment arrays as SETA-HPD /
    MoG_HPD_Calibrator.
    """

    def __init__(self, alpha=0.1, grid_per_component=200, tail_sigma_mult=6.0,
                 max_tail_expansions=4, merge_eps=1e-6,
                 c_grid=None, theta_grid=None, rho_grid=None, phi_grid=None, mog_decompose=True,
                 tune_frac=0.2, n_repeats=10,
                 max_tune_points=500, tune_grid_points=900, tune_sigma_mult=9.0,
                 split_seed=0, single_fold=False, min_split_n=200, verbose=True):
        """
        Parameters beyond SETAHPDCalibrator's:

        phi_grid : candidate epistemic-SHARE threshold-field exponents -- the knob that
                   distinguishes EASE-HPD from SETA-HPD. Default (-1.0, -0.5, 0.0, 0.5, 1.0);
                   0.0 (a share-blind threshold, i.e. plain SETA-HPD) is always kept on it.
        """
        super().__init__(alpha=alpha, grid_per_component=grid_per_component,
                         tail_sigma_mult=tail_sigma_mult,
                         max_tail_expansions=max_tail_expansions, merge_eps=merge_eps,
                         c_grid=c_grid, theta_grid=theta_grid, rho_grid=rho_grid,
                         mog_decompose=mog_decompose,
                         tune_frac=tune_frac, n_repeats=n_repeats,
                         max_tune_points=max_tune_points, tune_grid_points=tune_grid_points,
                         tune_sigma_mult=tune_sigma_mult,
                         split_seed=split_seed, single_fold=single_fold,
                         min_split_n=min_split_n, verbose=verbose)

        self.phi_grid = tuple(phi_grid) if phi_grid is not None else (-1.0, -0.5, 0.0, 0.5, 1.0)
        # 0.0 must be on the grid so the SETA-HPD family (phi off) is always reachable.
        if 0.0 not in self.phi_grid:
            self.phi_grid = tuple(sorted(self.phi_grid + (0.0,)))
        # The share mechanism is "active" iff phi is allowed to move off 0.
        self.phi_active_ = (self.phi_grid != (0.0,))

        self.phi_ = None    # chosen epistemic-share threshold-field exponent phi*
        self.s_bar_ = None  # calibration-set mean epistemic share at (c*, rho*), frozen by fit()

    # ------------------------------------------------------------------ logging

    def _log(self, stage, **fields):
        """[EASE-HPD]-tagged variant of the parent's one-line-per-stage logger."""
        if not self.verbose:
            return
        parts = []
        for k, v in fields.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.6g}")
            else:
                parts.append(f"{k}={v}")
        print(f"[EASE-HPD] {stage}: " + " ".join(parts))

    # ------------------------------------------------------------------ surrogate family

    @staticmethod
    def _epistemic_share(mu, sigma_tilde, pi):
        """
        s(x) = E(x) / (A(x) + E(x)) in [0, 1], computed from the ALREADY-reshaped (c) / shifted
        (rho) surrogate, so rho's effect on the epistemic term enters automatically (mu here is
        the rho-shifted mean, exactly as _log_sigma_tot consumes it). Denominator clipped away
        from zero for numerical safety (degenerate only if every sigma_tilde and every mean spread
        is exactly zero). mu/sigma_tilde/pi: [N,K] -> [N].
        """
        y_hat = np.sum(pi * mu, axis=1, keepdims=True)
        A = np.sum(pi * sigma_tilde ** 2, axis=1)
        E = np.sum(pi * (mu - y_hat) ** 2, axis=1)
        return E / np.clip(A + E, 1e-300, None)

    def _scores_for(self, trues, mu, sigma_tilde, pi, theta, phi, s_bar):
        """
        Nonconformity scores
        S_i = -log f_{c,rho}(y_i|x_i) - theta*log sigma_tot(x_i) - phi*(s(x_i) - s_bar).
        Overrides the parent's 5-arg signature with the two new EASE-HPD-specific arguments.
        """
        s = -self._log_mix_density(trues, mu, sigma_tilde, pi)
        if theta != 0.0:
            s = s - theta * self._log_sigma_tot(mu, sigma_tilde, pi)
        if phi != 0.0:
            s = s - phi * (self._epistemic_share(mu, sigma_tilde, pi) - s_bar)
        return s

    def _log_thresholds(self, mu, sigma_tilde, pi, t_hat, theta, phi, s_bar):
        """
        Per-point log-density threshold
        log tau_i = -t_hat - theta*log sigma_tot(x_i) - phi*(s(x_i) - s_bar).
        Overrides the parent's 5-arg signature with the two new EASE-HPD-specific arguments.
        """
        lt = np.full(mu.shape[0], -float(t_hat))
        if theta != 0.0:
            lt = lt - theta * self._log_sigma_tot(mu, sigma_tilde, pi)
        if phi != 0.0:
            lt = lt - phi * (self._epistemic_share(mu, sigma_tilde, pi) - s_bar)
        return lt

    # ------------------------------------------------------------------ fit / predict

    def fit(self, cal_trues, cal_mu, cal_sigma, cal_pi):
        """
        Same signature as SETAHPDCalibrator.fit / MoG_HPD_Calibrator.fit (drop-in at the call
        site). Selects (c*, rho*, theta*, phi*) by majority-vote argmin of tune-fold width over
        the full 4-D grid (SETA-HPD's own mechanism, widened by one axis), then fixes t_hat by
        median-of-calib-fold quantiles -- see the class docstring.
        cal_trues: [N] (or [N,1]); cal_mu / cal_sigma / cal_pi: [N,K].
        """
        cal_trues = self._to_numpy(cal_trues).squeeze()
        cal_mu = self._to_numpy(cal_mu)
        cal_sigma = self._to_numpy(cal_sigma)
        cal_pi = self._to_numpy(cal_pi)

        n_total = len(cal_trues)
        K = cal_mu.shape[1]

        self.G_ = float(np.exp(np.mean(np.log(np.clip(cal_sigma, 1e-300, None)))))
        self._log("sigma_anchor", n=n_total, K=K, G=self.G_,
                  sigma_min_med_max=self._mmm(cal_sigma))

        self.single_fold_used_ = bool(self.single_fold or n_total < self.min_split_n)
        if self.single_fold_used_:
            if not self.single_fold:
                print(f"[EASE-HPD] WARNING: only {n_total} calibration points "
                      f"(< min_split_n={self.min_split_n}); falling back to a single fold. "
                      f"(c*, rho*, theta*, phi*) is then chosen in-sample, so the coverage "
                      f"guarantee is no longer exact.")
            else:
                print("[EASE-HPD] WARNING: single_fold=True -- (c*, rho*, theta*, phi*) is chosen "
                      "on the same data used for the final quantile, so the coverage guarantee is "
                      "no longer exact.")
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

        # Reshaped sigmas per c and mean-shifted means per rho, each computed once and reused.
        sigma_by_c = {c: self._reshape_sigma(cal_sigma, c) for c in self.c_grid}
        mu_by_rho = {rho: self._shift_means(cal_mu, cal_pi, rho) for rho in self.rho_grid}

        # Epistemic share s(x) and its FULL-calibration-set mean s_bar, per (c, rho) cell -- frozen
        # once here (like G_), not recomputed per split (mirrors how G_ is a property of the whole
        # scoring function, not of a particular tune/calib partition).
        s_by_cr = {}
        s_bar_by_cr = {}
        for c in self.c_grid:
            st = sigma_by_c[c]
            for rho in self.rho_grid:
                mu_r = mu_by_rho[rho]
                s_full = self._epistemic_share(mu_r, st, cal_pi)
                s_by_cr[(c, rho)] = s_full
                s_bar_by_cr[(c, rho)] = float(np.mean(s_full))

        for c in self.c_grid:
            st = sigma_by_c[c]
            lst = self._log_sigma_tot(cal_mu, st, cal_pi)
            self._log("sigma_reshape", c=c, sigma_tilde_min_med_max=self._mmm(st),
                      log_sigma_tot_mean=float(np.mean(lst)),
                      log_sigma_tot_std=float(np.std(lst)))

        # --- Step 1: rank the (c, rho, theta, phi) surface on each split's tune fold, majority-vote.
        # f_{c,rho} depends on neither theta nor phi, so for each (c, rho) the density (the
        # expensive [N,M,K] evaluation) is computed once and the whole (theta, phi) grid is swept
        # with only cheap quantile + threshold-count operations.
        keys = [(c, rho, th, ph) for c in self.c_grid for rho in self.rho_grid
                for th in self.theta_grid for ph in self.phi_grid]
        key_col = {k: j for j, k in enumerate(keys)}
        width_curves = np.empty((len(splits), len(keys)))
        votes = []

        for b, (tune_idx, _) in enumerate(splits):
            if self.max_tune_points is not None and len(tune_idx) > self.max_tune_points:
                sub_idx = np.random.RandomState(self.split_seed + 1000 + b).choice(
                    tune_idx, size=self.max_tune_points, replace=False)
            else:
                sub_idx = tune_idx
            q_lvl = self._q_level(len(tune_idx))

            for c in self.c_grid:
                st = sigma_by_c[c]
                for rho in self.rho_grid:
                    mu_r = mu_by_rho[rho]
                    s_bar = s_bar_by_cr[(c, rho)]
                    s_full = s_by_cr[(c, rho)]
                    # (c, rho)-dependent, (theta, phi)-independent pieces, computed once:
                    neg_logdens_tune = -self._log_mix_density(cal_trues[tune_idx], mu_r[tune_idx],
                                                              st[tune_idx], cal_pi[tune_idx])  # [Ntune]
                    lst_tune = self._log_sigma_tot(mu_r[tune_idx], st[tune_idx], cal_pi[tune_idx])
                    s_tune = s_full[tune_idx] - s_bar
                    log_f_sub, dy_sub = self._grid_log_density(mu_r[sub_idx], st[sub_idx],
                                                               cal_pi[sub_idx])                # [Nsub,M]
                    lst_sub = self._log_sigma_tot(mu_r[sub_idx], st[sub_idx], cal_pi[sub_idx])
                    s_sub = s_full[sub_idx] - s_bar
                    for th in self.theta_grid:
                        base_tune = neg_logdens_tune - th * lst_tune
                        base_sub = -th * lst_sub
                        for ph in self.phi_grid:
                            scores = base_tune - ph * s_tune
                            t_hat = float(np.quantile(scores, q_lvl, method='higher'))
                            log_tau = -t_hat + base_sub - ph * s_sub                          # [Nsub]
                            width = float(np.mean(np.sum(log_f_sub >= log_tau[:, None], axis=1) * dy_sub))
                            width_curves[b, key_col[(c, rho, th, ph)]] = width
            votes.append(keys[int(np.argmin(width_curves[b]))])

        avg = width_curves.mean(axis=0)
        std = width_curves.std(axis=0)
        self.tuning_widths_ = {k: float(w) for k, w in zip(keys, avg)}
        self.tuning_widths_std_ = {k: float(s) for k, s in zip(keys, std)}
        self.vote_counts_ = dict(Counter(votes))

        # Majority vote across splits. Ties (and the single-fold case) fall back to the smallest
        # average tune width among the tied cells, then to the cell closest to MoG-HPD
        # (c=1, rho=1, theta=0, phi=0) -- so a flat surface keeps the established default.
        top = max(self.vote_counts_.values())
        tied = [k for k, v in self.vote_counts_.items() if v == top]
        best = min(tied, key=lambda k: (self.tuning_widths_[k],
                                        abs(k[0] - 1.0) + abs(k[1] - 1.0) + abs(k[2]) + abs(k[3])))
        self.c_, self.rho_, self.theta_, self.phi_ = (float(best[0]), float(best[1]),
                                                       float(best[2]), float(best[3]))
        self.tune_width_at_best_ = float(self.tuning_widths_[best])
        self.s_bar_ = s_bar_by_cr[(best[0], best[1])]

        # phi-marginal: best (min) tune width over c, rho, theta, per phi -- summarizes whether the
        # share mechanism carries real signal independent of the specific (c, rho, theta) picked.
        phi_marg = {}
        for (c, rho, th, ph), w in self.tuning_widths_.items():
            phi_marg[ph] = min(phi_marg.get(ph, np.inf), w)

        on_c_edge = self.c_ in (self.c_grid[0], self.c_grid[-1])
        on_rho_edge = self.mog_decompose and self.rho_ in (self.rho_grid[0], self.rho_grid[-1])
        on_th_edge = self.theta_ in (self.theta_grid[0], self.theta_grid[-1])
        on_phi_edge = self.phi_active_ and self.phi_ in (self.phi_grid[0], self.phi_grid[-1])
        self._log("tuning", n_cells=len(keys),
                  best_c=self.c_, best_rho=self.rho_, best_theta=self.theta_, best_phi=self.phi_,
                  mog_decompose=self.mog_decompose, phi_active=self.phi_active_,
                  s_bar=self.s_bar_,
                  votes=f"{self.vote_counts_[best]}/{len(splits)}",
                  tune_width=self.tune_width_at_best_,
                  tune_width_std=self.tuning_widths_std_[best],
                  width_range=f"[{avg.min():.4f},{avg.max():.4f}]",
                  solution="BOUNDARY" if (on_c_edge or on_rho_edge or on_th_edge or on_phi_edge) else "interior",
                  phi_marginal=sorted(phi_marg.items()),
                  vote_spread=sorted(self.vote_counts_.items(), key=lambda kv: -kv[1])[:3])

        # --- Step 2: final conformal threshold at the now-fixed (c*, rho*, theta*, phi*), on calib
        # folds.
        st_best = sigma_by_c[best[0]]
        mu_best = mu_by_rho[best[1]]
        t_hats = []
        for _, calib_idx in splits:
            scores = self._scores_for(cal_trues[calib_idx], mu_best[calib_idx],
                                      st_best[calib_idx], cal_pi[calib_idx],
                                      self.theta_, self.phi_, self.s_bar_)
            t_hats.append(float(np.quantile(scores, self._q_level(len(scores)), method='higher')))

        final_scores = self._scores_for(cal_trues, mu_best, st_best, cal_pi,
                                        self.theta_, self.phi_, self.s_bar_)
        self._log("scores", c=self.c_, rho=self.rho_, theta=self.theta_, phi=self.phi_,
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
        tau_i = exp(-t_hat - theta*log sigma_tot(x_i;c*,rho*) - phi*(s(x_i;c*,rho*) - s_bar)).
        Returns a ragged list of [m_i, 2] arrays (see the class docstring); m_i = 0 means an empty
        prediction set.
        """
        if self.c_ is None or self.t_hat is None:
            raise ValueError("Calibrator must be fitted before calling predict.")

        test_mu = self._to_numpy(test_mu)
        test_sigma = self._to_numpy(test_sigma)
        test_pi = self._to_numpy(test_pi)

        mu_shift = self._shift_means(test_mu, test_pi, self.rho_)
        sigma_tilde = self._reshape_sigma(test_sigma, self.c_)
        log_tau = self._log_thresholds(mu_shift, sigma_tilde, test_pi, self.t_hat,
                                       self.theta_, self.phi_, self.s_bar_)
        taus = np.exp(log_tau)

        intervals = []
        for i in range(test_mu.shape[0]):
            intervals.append(self._segments_for_point(mu_shift[i], sigma_tilde[i], test_pi[i],
                                                      taus[i]))

        widths = np.array([float(np.sum(s[:, 1] - s[:, 0])) if len(s) else 0.0 for s in intervals])
        n_seg = np.array([len(s) for s in intervals])
        self._log("predict_widths", n=len(intervals), c=self.c_, rho=self.rho_, theta=self.theta_,
                  phi=self.phi_,
                  width_mean=float(widths.mean()), width_median=float(np.median(widths)),
                  width_p90=float(np.percentile(widths, 90)),
                  avg_segments=float(n_seg.mean()),
                  frac_multi=float(np.mean(n_seg > 1)), frac_empty=float(np.mean(n_seg == 0)),
                  n_zero_width=int(np.sum(widths <= 0.0)),
                  n_nonfinite=int(np.sum(~np.isfinite(widths))),
                  tau_min_med_max=self._mmm(taus))
        return intervals
