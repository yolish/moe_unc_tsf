import numpy as np
from collections import Counter

from calibration.tabular_regression.SETA_HPD_Calibrator import SETAHPDCalibrator


class MELDHPDCalibrator(SETAHPDCalibrator):
    """
    MELD-HPD -- Maximum-likelihood Epistemic-aLeatoric Decomposition HPD conformal calibration
    for Mixture-of-Gaussians models.

    MELD-HPD uses the exact same surrogate family as SETA-HPD (SETAHPDCalibrator): it reshapes
    BOTH terms of the Mixture-of-Gaussians law-of-total-variance decomposition
        Var(Y|x) = sum_k w_k sigma_k^2      (within-component / aleatoric)
                 + sum_k w_k (mu_k - yhat)^2 (between-component / epistemic),
    the aleatoric half by a within-component shape exponent c
    (tilde_sigma_k = G (sigma_k/G)^c) and the epistemic half by a between-component scale
    exponent rho via the mean-shift tilde_mu_k = yhat + sqrt(rho)(mu_k - yhat), then thresholds
    the resulting surrogate density f_{c,rho} with a threshold field theta -- identical machinery
    to SETA-HPD, whose helpers and predictor this class inherits unchanged.

    What differs from SETA-HPD is ONLY how (c, rho, theta) are selected.
    -----------------------------------------------------------------------------------------
    STA-HPD and SETA-HPD pick their exponents by argmin of the raw tune-fold interval WIDTH.
    Width on a subsampled tune fold is a noisy objective, and adding SETA-HPD's epistemic
    exponent rho gives that argmin more room to fit the noise -- so the chosen rho* does not
    always generalize, and SETA-HPD can come out WIDER than STA-HPD on held-out test data
    (documented in docs/sta_hpd_mog_decomposition.md).

    MELD-HPD instead selects the two SHAPE exponents (c, rho) -- the ones that define the density
    f_{c,rho} -- by a PROPER SCORING RULE: it maximizes the mean calibration log predictive
    density on the tune fold,

        L(c, rho) = mean_i log f_{c,rho}(y_i | x_i).

    Maximizing L is exactly minimizing KL(true conditional || f_{c,rho}): it selects the reshaping
    whose surrogate density best matches the data. By the Neyman-Pearson width-optimality of
    highest-density regions (MoG_HPD_Calibrator's docstring / the STA-HPD paper, Thm. 1), the HPD
    of the best-matching density is the narrowest VALID set, so a better-matched density yields a
    narrower conformalized HPD. The log score is strictly proper, smooth, and evaluated on ALL
    tune points (not a noisy width count on a subsample), so it does not chase width noise -- rho
    stops being grid-lucked and becomes a quantity derived from the density's shape. Concretely:
    when experts are over-separated (the true outcome sits near yhat) the log-density is higher at
    rho < 1, so a compressed epistemic term is selected and the interval narrows; when the truth
    is genuinely multimodal, L peaks near rho = 1 and the modes are kept.

    The threshold field theta does NOT enter f_{c,rho} (it only tilts the per-point threshold, not
    the density), so a density-based score cannot rank it. theta is therefore still selected by
    argmin tune-fold width at the fixed (c*, rho*) -- its proper objective, and a single
    well-behaved dimension far less prone to overfitting than the old joint (c, rho, theta) width
    argmin.

    As an interpretable anchor (a diagnostic, NOT the decision) MELD-HPD also reports a
    closed-form epistemic-reliability estimate: with A_i = sum_k w_k tilde_sigma_k^2(c*),
    E_i = sum_k w_k (mu_k - yhat)^2 and squared residuals R_i^2 = (y_i - yhat_i)^2,

        rho_hat_closed = clip( sum_i E_i (R_i^2 - A_i) / sum_i E_i^2 , 0, None ),

    the least-squares fit of the reshaped total variance A + rho*E to the squared residuals --
    "how much of the reported epistemic spread the residuals actually support."

    Conformal validity
    ------------------
    Unchanged from SETA-HPD. (c, rho, theta) are selected using only the tune folds; the final
    t_hat is the median of the per-split calib-fold quantiles at the now-fixed (c*, rho*, theta*),
    on folds disjoint from the tune folds. The score is therefore a frozen, label-independent
    function by the time it is conformalized, so the finite-sample split-conformal marginal
    guarantee P(y in C(x)) >= 1 - alpha holds exactly, with no assumption that the mixture is well
    specified -- only the width depends on model quality.

    predict() is inherited verbatim from SETAHPDCalibrator (it depends only on c_, rho_, theta_,
    t_hat), so the returned object is the same ragged list of [m_i, 2] disjoint-segment arrays.
    """

    def _log(self, stage, **fields):
        """[MELD-HPD]-tagged variant of the parent's one-line-per-stage logger."""
        if not self.verbose:
            return
        parts = []
        for k, v in fields.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.6g}")
            else:
                parts.append(f"{k}={v}")
        print(f"[MELD-HPD] {stage}: " + " ".join(parts))

    @staticmethod
    def _closed_form_rho(trues, mu, sigma_tilde, pi):
        """
        Least-squares epistemic-reliability estimate rho_hat (diagnostic only):
        with A_i = sum_k pi_k tilde_sigma_k^2, E_i = sum_k pi_k (mu_k - yhat)^2 and
        R_i^2 = (y_i - yhat_i)^2, rho_hat minimizes sum_i (A_i + rho E_i - R_i^2)^2, clipped to
        >= 0. Returns np.nan when the epistemic term is identically zero (e.g. K = 1), where rho
        has no effect at all. mu/sigma_tilde/pi: [N,K]; trues: [N].
        """
        y_hat = np.sum(pi * mu, axis=1)                                   # [N]
        A = np.sum(pi * sigma_tilde ** 2, axis=1)                         # [N] aleatoric
        E = np.sum(pi * (mu - y_hat[:, None]) ** 2, axis=1)               # [N] epistemic
        R2 = (np.asarray(trues).squeeze() - y_hat) ** 2                   # [N] squared residual
        denom = float(np.sum(E ** 2))
        if denom <= 1e-30:
            return float('nan')
        rho_hat = float(np.sum(E * (R2 - A)) / denom)
        return max(rho_hat, 0.0)

    def fit(self, cal_trues, cal_mu, cal_sigma, cal_pi):
        """
        Same signature as SETAHPDCalibrator.fit / MoG_HPD_Calibrator.fit (drop-in at the call
        site). Selects (c*, rho*) by max mean tune-fold log-density and theta* by min tune-fold
        width, then fixes t_hat by median-of-calib-fold quantiles -- see the class docstring.
        cal_trues: [N] (or [N,1]); cal_mu / cal_sigma / cal_pi: [N,K].
        """
        cal_trues = self._to_numpy(cal_trues).squeeze()
        cal_mu = self._to_numpy(cal_mu)
        cal_sigma = self._to_numpy(cal_sigma)
        cal_pi = self._to_numpy(cal_pi)

        n_total = len(cal_trues)
        K = cal_mu.shape[1]

        # Freeze the sigma anchor on the FULL calibration set, before any fold is drawn.
        self.G_ = float(np.exp(np.mean(np.log(np.clip(cal_sigma, 1e-300, None)))))
        self._log("sigma_anchor", n=n_total, K=K, G=self.G_,
                  sigma_min_med_max=self._mmm(cal_sigma))

        self.single_fold_used_ = bool(self.single_fold or n_total < self.min_split_n)
        if self.single_fold_used_:
            if not self.single_fold:
                print(f"[MELD-HPD] WARNING: only {n_total} calibration points "
                      f"(< min_split_n={self.min_split_n}); falling back to a single fold. "
                      f"(c*, rho*, theta*) is then chosen in-sample, so the coverage guarantee is no "
                      f"longer exact.")
            else:
                print("[MELD-HPD] WARNING: single_fold=True -- (c*, rho*, theta*) is chosen on the "
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

        # Reshaped sigmas per c and mean-shifted means per rho, each computed once and reused.
        sigma_by_c = {c: self._reshape_sigma(cal_sigma, c) for c in self.c_grid}
        mu_by_rho = {rho: self._shift_means(cal_mu, cal_pi, rho) for rho in self.rho_grid}

        # --- Step 1a: rank the (c, rho) SHAPE cells by mean tune-fold LOG-DENSITY (proper score),
        # and also fill the full (c, rho, theta) WIDTH surface (for theta selection + diagnostics).
        cr_keys = [(c, rho) for c in self.c_grid for rho in self.rho_grid]
        cr_col = {k: j for j, k in enumerate(cr_keys)}
        keys = [(c, rho, th) for c in self.c_grid for rho in self.rho_grid
                for th in self.theta_grid]
        key_col = {k: j for j, k in enumerate(keys)}

        logscore_curves = np.empty((len(splits), len(cr_keys)))   # mean log-density per (c,rho)
        width_curves = np.empty((len(splits), len(keys)))         # tune width per (c,rho,theta)
        shape_votes = []

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
                    # (c, rho)-dependent, theta-independent pieces, computed once:
                    log_f_true = self._log_mix_density(cal_trues[tune_idx], mu_r[tune_idx],
                                                       st[tune_idx], cal_pi[tune_idx])   # [Ntune]
                    # Proper score = mean log-density on the WHOLE tune fold. Clip the per-point
                    # floor so a single density-underflow outlier cannot send the cell to -inf;
                    # -1e3 nats is astronomically unlikely, so ordering among real cells is intact.
                    logscore_curves[b, cr_col[(c, rho)]] = float(np.mean(np.maximum(log_f_true, -1e3)))

                    neg_logdens_tune = -log_f_true                                       # [Ntune]
                    lst_tune = self._log_sigma_tot(mu_r[tune_idx], st[tune_idx], cal_pi[tune_idx])
                    log_f_sub, dy_sub = self._grid_log_density(mu_r[sub_idx], st[sub_idx],
                                                               cal_pi[sub_idx])           # [Nsub,M]
                    lst_sub = self._log_sigma_tot(mu_r[sub_idx], st[sub_idx], cal_pi[sub_idx])
                    for th in self.theta_grid:
                        scores = neg_logdens_tune - th * lst_tune
                        t_hat = float(np.quantile(scores, q_lvl, method='higher'))
                        log_tau = -t_hat - th * lst_sub                                  # [Nsub]
                        width = float(np.mean(np.sum(log_f_sub >= log_tau[:, None], axis=1) * dy_sub))
                        width_curves[b, key_col[(c, rho, th)]] = width
            shape_votes.append(cr_keys[int(np.argmax(logscore_curves[b]))])

        avg_logscore = logscore_curves.mean(axis=0)
        std_logscore = logscore_curves.std(axis=0)
        avg_width = width_curves.mean(axis=0)
        std_width = width_curves.std(axis=0)
        self.logscore_surface_ = {k: float(v) for k, v in zip(cr_keys, avg_logscore)}
        self.tuning_widths_ = {k: float(w) for k, w in zip(keys, avg_width)}
        self.tuning_widths_std_ = {k: float(s) for k, s in zip(keys, std_width)}

        # --- Step 1b: (c*, rho*) = majority vote of the per-split log-density argmaxes. Ties (and
        # the single-fold case) fall back to the largest mean log-density among tied cells, then to
        # the cell closest to MoG-HPD's (c=1, rho=1), so a flat density surface keeps the default.
        shape_counts = dict(Counter(shape_votes))
        top_shape = max(shape_counts.values())
        tied_shape = [k for k, v in shape_counts.items() if v == top_shape]
        best_cr = max(tied_shape, key=lambda k: (self.logscore_surface_[k],
                                                 -(abs(k[0] - 1.0) + abs(k[1] - 1.0))))
        c_star, rho_star = float(best_cr[0]), float(best_cr[1])

        # --- Step 1c: theta* = majority vote of the per-split width argmins AT (c*, rho*). Ties
        # fall back to the smallest mean width, then to theta closest to 0 (a single global
        # threshold), mirroring SETA-HPD's tie-break toward the MoG-HPD corner.
        theta_votes = []
        for b in range(len(splits)):
            widths_b = [width_curves[b, key_col[(c_star, rho_star, th)]] for th in self.theta_grid]
            theta_votes.append(self.theta_grid[int(np.argmin(widths_b))])
        theta_counts = dict(Counter(theta_votes))
        top_theta = max(theta_counts.values())
        tied_theta = [th for th, v in theta_counts.items() if v == top_theta]
        theta_star = min(tied_theta, key=lambda th: (self.tuning_widths_[(c_star, rho_star, th)],
                                                     abs(th)))

        self.c_, self.rho_, self.theta_ = c_star, rho_star, float(theta_star)
        best = (self.c_, self.rho_, self.theta_)
        self.tune_width_at_best_ = float(self.tuning_widths_[best])
        # vote_counts_ is reported by the exp-layer diagnostics as (c,rho,theta) cells; combine the
        # shape vote and theta vote into the chosen cell's tally for a comparable summary.
        self.vote_counts_ = {(c_star, rho_star, th): theta_counts.get(th, 0) for th in self.theta_grid}
        self.vote_counts_[best] = min(shape_counts[best_cr], theta_counts.get(theta_star, 0))

        # Closed-form epistemic-reliability anchor at the chosen c* (diagnostic only).
        self.rho_hat_closed_ = self._closed_form_rho(cal_trues, cal_mu, sigma_by_c[c_star], cal_pi)

        on_c_edge = self.c_ in (self.c_grid[0], self.c_grid[-1])
        on_rho_edge = self.mog_decompose and self.rho_ in (self.rho_grid[0], self.rho_grid[-1])
        on_th_edge = self.theta_ in (self.theta_grid[0], self.theta_grid[-1])
        self._log("tuning", n_cells=len(keys), objective="logdensity(c,rho)+width(theta)",
                  best_c=self.c_, best_rho=self.rho_, best_theta=self.theta_,
                  mog_decompose=self.mog_decompose,
                  shape_votes=f"{shape_counts[best_cr]}/{len(splits)}",
                  theta_votes=f"{theta_counts.get(theta_star, 0)}/{len(splits)}",
                  logscore_at_best=self.logscore_surface_[best_cr],
                  logscore_std=float(std_logscore[cr_col[best_cr]]),
                  rho_hat_closed=self.rho_hat_closed_,
                  tune_width=self.tune_width_at_best_,
                  logscore_range=f"[{avg_logscore.min():.4f},{avg_logscore.max():.4f}]",
                  solution="BOUNDARY" if (on_c_edge or on_rho_edge or on_th_edge) else "interior",
                  shape_vote_spread=sorted(shape_counts.items(), key=lambda kv: -kv[1])[:3])

        # --- Step 2: final conformal threshold at the now-fixed (c*, rho*, theta*), on calib folds.
        st_best = sigma_by_c[c_star]
        mu_best = mu_by_rho[rho_star]
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
