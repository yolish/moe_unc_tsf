import numpy as np
from scipy.special import logsumexp
from scipy.optimize import brentq


class MoG_HPD_Calibrator:
    """
    Highest-density-region (HPD) conformal calibrator for Mixture-of-Gaussians (MoG) models.

    Unlike the other calibrators in this package, this one does not conformalize a scalar
    variance decomposition around a symmetric ŷ ± s(x) interval. Instead it uses the model's
    own closed-form mixture density f_mix(y|x) = sum_k pi_k(x) * N(y; mu_k(x), sigma_k(x)^2)
    directly: by a Neyman-Pearson-style argument, the width-minimal set achieving a fixed
    coverage level for a given density is its super-level ("highest density") set. The
    nonconformity score S_i = -log f_mix(y_i|x_i) is conformalized with the same finite-sample
    quantile correction used by every other calibrator here, giving the standard split-conformal
    marginal coverage guarantee regardless of whether the fitted mixture is well specified —
    only the resulting width depends on how good the fit is.

    The induced prediction set {y : f_mix(y|x) >= tau} is not required to be a single interval:
    for well-separated modes it can be a union of up to K disjoint intervals. predict() returns
    a ragged list of per-point segment arrays rather than a single [N,2] interval array.
    """

    def __init__(self, alpha=0.1, grid_per_component=200, tail_sigma_mult=6.0,
                 max_tail_expansions=4, merge_eps=1e-6):
        self.alpha = alpha
        self.grid_per_component = grid_per_component
        self.tail_sigma_mult = tail_sigma_mult
        self.max_tail_expansions = max_tail_expansions
        self.merge_eps = merge_eps

        self.t_hat = None   # calibrated threshold on S = -log f_mix (nats)
        self.tau_ = None     # exp(-t_hat), the same threshold in density space

    @staticmethod
    def _to_numpy(x):
        return x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)

    def _log_mix_density(self, y, mu, sigma, pi):
        """
        y: [N], mu/sigma/pi: [N,K] (one calibration/test point per row). Returns
        log f_mix(y|x) : [N]. Mirrors the exact mixture log-likelihood used at training time
        (Exp_Tabular_Regression._calculate_loss, prob_expert & not unc_gating branch).
        """
        y = np.asarray(y)
        log_pi = np.log(np.clip(pi, 1e-12, None))
        log_comp = (-0.5 * np.log(2 * np.pi) - np.log(sigma)
                    - 0.5 * ((y[:, None] - mu) / sigma) ** 2)
        return logsumexp(log_pi + log_comp, axis=1)

    @staticmethod
    def _log_mix_density_1d(y, mu, sigma, pi):
        """
        Single test point's mixture, evaluated at an arbitrary array of y values.
        y: [M], mu/sigma/pi: [K] (shared across all M query points). Returns log f_mix(y) : [M].
        """
        log_pi = np.log(np.clip(pi, 1e-12, None))            # [K]
        log_comp = (-0.5 * np.log(2 * np.pi) - np.log(sigma)
                    - 0.5 * ((y[:, None] - mu[None, :]) / sigma[None, :]) ** 2)  # [M,K]
        return logsumexp(log_comp + log_pi[None, :], axis=1)  # [M]

    def _q_level(self, n):
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        return min(max(q_level, 0.0), 1.0)

    def fit(self, cal_trues, cal_mu, cal_sigma, cal_pi):
        cal_trues = self._to_numpy(cal_trues).squeeze()
        cal_mu = self._to_numpy(cal_mu)
        cal_sigma = self._to_numpy(cal_sigma)
        cal_pi = self._to_numpy(cal_pi)

        log_f = self._log_mix_density(cal_trues, cal_mu, cal_sigma, cal_pi)
        scores = -log_f

        n = len(scores)
        q_level = self._q_level(n)
        self.t_hat = float(np.quantile(scores, q_level, method='higher'))
        self.tau_ = float(np.exp(-self.t_hat))

    def _segments_for_point(self, mu, sigma, pi, tau):
        """
        Find the (possibly disjoint) set of y-intervals where f_mix(y) >= tau for one test
        point's mixture parameters mu/sigma/pi (each [K]). Returns a [m,2] numpy array.
        """
        def f(y_arr):
            return np.exp(self._log_mix_density_1d(np.atleast_1d(y_arr), mu, sigma, pi))

        lo_win = float(np.min(mu - self.tail_sigma_mult * sigma))
        hi_win = float(np.max(mu + self.tail_sigma_mult * sigma))

        expansions = 0
        while expansions < self.max_tail_expansions:
            edge_vals = f(np.array([lo_win, hi_win]))
            if np.all(edge_vals <= max(tau * 1e-3, 1e-300)):
                break
            margin = (hi_win - lo_win) * 0.5
            lo_win -= margin
            hi_win += margin
            expansions += 1

        # Per-component fine grid (resolution scaled to that component's own sigma) plus a
        # coarser bridging grid across the full window, to catch crossings between components.
        grid_parts = [np.linspace(lo_win, hi_win, 3 * self.grid_per_component)]
        for k in range(len(mu)):
            grid_parts.append(np.linspace(mu[k] - self.tail_sigma_mult * sigma[k],
                                           mu[k] + self.tail_sigma_mult * sigma[k],
                                           self.grid_per_component))
        grid = np.unique(np.concatenate(grid_parts))
        grid = grid[(grid >= lo_win) & (grid <= hi_win)]

        g = f(grid) - tau
        sign = np.sign(g)
        # Treat exact zeros as a tiny positive nudge so sign changes are still detected cleanly.
        sign[sign == 0] = 1e-12
        change_idx = np.where(np.diff(np.sign(sign)) != 0)[0]

        roots = []
        for idx in change_idx:
            y_a, y_b = grid[idx], grid[idx + 1]
            try:
                root = brentq(lambda y: float(f(np.array([y]))[0]) - tau, y_a, y_b, xtol=1e-10, maxiter=200)
                roots.append(root)
            except ValueError:
                continue
        roots = sorted(roots)

        if len(roots) % 2 != 0:
            # Should not happen given f(lo_win), f(hi_win) < tau by construction; drop a
            # duplicate/degenerate root rather than silently mis-pairing the rest.
            roots = roots[:-1]

        if len(roots) == 0:
            if float(f(np.array([(lo_win + hi_win) / 2]))[0]) >= tau or np.max(f(grid)) >= tau:
                # Peak clears tau but the coarse root search missed it (shouldn't happen with
                # the windowing above) — fall back to the full window as a conservative segment.
                return np.array([[lo_win, hi_win]])
            return np.zeros((0, 2))

        segments = np.array(roots).reshape(-1, 2)

        # Merge segments whose gap is smaller than merge_eps (grid/bisection tolerance noise).
        merged = [segments[0]]
        for seg in segments[1:]:
            if seg[0] - merged[-1][1] < self.merge_eps:
                merged[-1][1] = seg[1]
            else:
                merged.append(seg)
        return np.array(merged)

    def predict(self, test_mu, test_sigma, test_pi):
        if self.tau_ is None:
            raise ValueError("Calibrator must be fitted before calling predict.")

        test_mu = self._to_numpy(test_mu)
        test_sigma = self._to_numpy(test_sigma)
        test_pi = self._to_numpy(test_pi)

        intervals = []
        for i in range(test_mu.shape[0]):
            intervals.append(self._segments_for_point(test_mu[i], test_sigma[i], test_pi[i], self.tau_))
        return intervals
