"""CP-DVS: Conformal Prediction via Decomposed Variance Scaling.

A variance-scaled conformal calibrator for Mixture-of-Gaussians (MoG) forecasters.
Where CP-VS normalizes the residual by the single scalar sigma_total(x), CP-DVS keeps
the Law-of-Total-Variance decomposition of the mixture intact and *learns* how the
interval half-width should depend on its two components.

Setting
-------
A MoG head emits {pi_k(x), mu_k(x), sigma_k^2(x)}_{k=1..K} per point. The law of total
variance splits the predictive variance into

    mu_bar(x)          = sum_k pi_k mu_k                                (mixture mean)
    sigma^2_within(x)  = sum_k pi_k sigma_k^2                           (within-component)
    sigma^2_between(x) = sum_k pi_k (mu_k - mu_bar)^2                   (between-component)
    sigma^2_total(x)   = sigma^2_within(x) + sigma^2_between(x)

Method
------
CP-VS fixes the non-conformity score to |y - mu_bar| / sigma_total. That choice is only
width-optimal when sigma_total is proportional to the conditional (1-alpha) quantile of
the absolute residual. It rarely is: sigma_total is a conditional *standard deviation*,
and the two variance components carry different amounts of information about the residual
scale (the between-component measures expert disagreement, which on a well-fit mixture is
a much weaker predictor of error than the within-component aleatoric noise).

CP-DVS instead scores against a learned scale field

    log u(x) = beta_0 + beta_1 * l(x) + beta_2 * l(x)^2 + beta_3 * d(x) + beta_4 * rho(x)

    l(x)   = log sigma_total(x), centered on the fit split
    d(x)   = log sigma_within(x) - log sigma_total(x)   in (-inf, 0]  (component mix)
    rho(x) = sigma^2_between(x) / sigma^2_total(x)      in [0, 1]     (epistemic share)

with beta fit by *quantile regression at the target level* tau = 1 - alpha on
log|y - mu_bar|. Because log is monotone, the tau-quantile of log|r| exponentiates to the
tau-quantile of |r|, so exp(z^T beta) estimates the conditional (1-alpha) quantile of the
absolute residual directly -- the quantity whose value *is* the oracle half-width. The
pinball objective is convex in beta, so this is a well-posed fit rather than a grid search.

Then the usual split-conformal step restores finite-sample validity:

    s_i    = |y_i - mu_bar_i| / u(x_i)
    q_hat  = Quantile_{ceil((n+1)(1-alpha))/n}({s_i})
    C(x)   = mu_bar(x) +/- q_hat * u(x)

so coverage does not depend on the scale model being right; only the width does.

Why this is expected to be tighter
----------------------------------
Two baselines are exact points of the beta family, and both are evaluated explicitly
during the fit (see `_fit_scale_model`), with the returned beta being the best of the
three by pinball loss:

    beta = (c, 0, 0, 0, 0)  ->  u = const       ->  Standard CP
    beta = (c, 1, 0, 0, 0)  ->  u = sigma_total ->  CP-VS

So on the fitting objective CP-DVS is never worse than either baseline by construction.
The fit objective is the conditional-quantile pinball loss, and the width of a
variance-scaled conformal band is governed by E[u(X)] * q_hat with q_hat -> 1 as u
approaches the true conditional quantile -- which is exactly what the pinball loss
measures. The remaining gap between "better on the fit objective" and "narrower on test"
is generalization, which the benchmark measures rather than assumes.

Conformal quantiles are taken per (horizon step, channel) -- axis 0 of the score array --
matching the convention of the other time-series calibrators in this package. The
per-(h, c) quantile absorbs any factor of u that is constant within a (h, c) cell, so the
scale model only has to explain the *within-cell* heteroscedasticity across time.
"""

import numpy as np

EPS = 1e-12
_LOG_FLOOR = -30.0
_D_FLOOR = -12.0


def decompose_mog(pi, mu, var):
    """Law-of-total-variance decomposition of a MoG head.

    Parameters
    ----------
    pi, mu, var : array_like, shape [..., K, ...]
        Mixture weights, component means and component *variances*, stacked along
        `axis=1` (the layout the MoE model in this repo emits: [N, K, H, C]).

    Returns
    -------
    mu_bar, v_within, v_between, v_total : ndarray, shape [N, H, C]
    """
    pi = np.asarray(pi, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    var = np.maximum(np.asarray(var, dtype=np.float64), 0.0)

    # Guard against gates that do not sum to one (numerical drift in the softmax).
    pi = pi / np.maximum(pi.sum(axis=1, keepdims=True), EPS)

    mu_bar = np.sum(pi * mu, axis=1)
    v_within = np.sum(pi * var, axis=1)
    v_between = np.sum(pi * (mu - mu_bar[:, None]) ** 2, axis=1)
    v_total = v_within + v_between
    return mu_bar, v_within, v_between, v_total


class MoGPrediction:
    """Container for one split's MoG head outputs, with the decomposition cached.

    Holding the mixture parameters (rather than a pre-reduced sigma) is what lets a
    calibrator see the within/between split at all; `pi`, `mu`, `var` are [N, K, H, C].
    """

    __slots__ = ("pi", "mu", "var", "mu_bar", "v_within", "v_between", "v_total")

    def __init__(self, pi, mu, var):
        self.pi, self.mu, self.var = pi, mu, var
        self.mu_bar, self.v_within, self.v_between, self.v_total = decompose_mog(pi, mu, var)

    @classmethod
    def from_decomposition(cls, mu_bar, v_within, v_between):
        """Build from pre-reduced components, keeping the per-expert arrays unmaterialized.

        Every calibrator here consumes only the decomposition, and for long horizons the
        [N, K, H, C] tensors are an order of magnitude larger than the [N, H, C] ones, so
        the reduction is worth doing at the point of inference. `pi`/`mu`/`var` are left
        as None; anything needing the full mixture density (e.g. an HPD set) must use the
        regular constructor.
        """
        obj = cls.__new__(cls)
        obj.pi = obj.mu = obj.var = None
        obj.mu_bar = np.asarray(mu_bar, dtype=np.float64)
        obj.v_within = np.maximum(np.asarray(v_within, dtype=np.float64), 0.0)
        obj.v_between = np.maximum(np.asarray(v_between, dtype=np.float64), 0.0)
        obj.v_total = obj.v_within + obj.v_between
        return obj

    @property
    def shape(self):
        return self.mu_bar.shape

    def subset(self, idx):
        """Slice along the sample axis, returning a new MoGPrediction."""
        if self.pi is None:  # built via from_decomposition; slice the components directly
            return MoGPrediction.from_decomposition(
                self.mu_bar[idx], self.v_within[idx], self.v_between[idx])
        return MoGPrediction(self.pi[idx], self.mu[idx], self.var[idx])


def conformal_quantile_level(n, alpha):
    """Finite-sample split-conformal level ceil((n+1)(1-alpha))/n, clipped to [0, 1]."""
    if n <= 0:
        raise ValueError("Cannot take a conformal quantile of an empty calibration set.")
    return float(min(max(np.ceil((n + 1) * (1.0 - alpha)) / n, 0.0), 1.0))


def _pinball(residual, tau):
    """Mean pinball loss at level tau for the signed errors `residual = t - prediction`."""
    return float(np.mean(residual * (tau - (residual < 0.0))))


class CPDVSCalibrator:
    """Conformal Prediction via Decomposed Variance Scaling.

    Parameters
    ----------
    alpha : float
        Miscoverage level; intervals target 1 - alpha coverage.
    fit_frac : float
        Fraction of the calibration data (leading, in time order) spent fitting the scale
        model. The remainder produces the conformal quantile. The two must be disjoint or
        the quantile is optimistically biased and coverage is no longer guaranteed.
    quadratic : bool
        Include the l^2 feature, which lets the scale bend as a power of sigma_total
        rather than tracking it linearly in log space.
    n_iter, lr : int, float
        Full-batch Adam budget for the quantile regression.
    max_fit_rows : int
        Cap on rows used by the quantile regression (strided, deterministic). The fit has
        <=5 parameters, so a few hundred thousand rows is far past saturation.
    ridge : float
        L2 penalty on the non-intercept coefficients; guards the fit when a feature is
        near-degenerate (e.g. K=1, where the between-component is identically zero).
    """

    def __init__(self, alpha=0.1, fit_frac=0.5, quadratic=True,
                 n_iter=1500, lr=0.05, max_fit_rows=400_000, ridge=1e-4):
        self.alpha = alpha
        self.fit_frac = fit_frac
        self.quadratic = quadratic
        self.n_iter = n_iter
        self.lr = lr
        self.max_fit_rows = max_fit_rows
        self.ridge = ridge

        # Learned state.
        self.beta = None
        self.q_hat = None
        self._l_mean = None
        self._z_scale = None
        self.fit_report_ = {}

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------

    def _features(self, pred):
        """Build the design matrix Z with shape [N, H, C, P] from a MoGPrediction."""
        v_within = np.maximum(pred.v_within, 0.0)
        v_total = np.maximum(pred.v_total, EPS)

        l = 0.5 * np.log(np.maximum(v_total, EPS))
        l = np.maximum(l, _LOG_FLOOR)
        if self._l_mean is None:
            self._l_mean = float(np.mean(l))
        lc = l - self._l_mean

        # d = log sigma_within - log sigma_total <= 0; -> 0 when the mixture is degenerate
        # (all mass in the within term), very negative when experts disagree strongly.
        d = 0.5 * (np.log(np.maximum(v_within, EPS)) - np.log(v_total))
        d = np.clip(d, _D_FLOOR, 0.0)

        rho = np.clip(pred.v_between / v_total, 0.0, 1.0)

        cols = [np.ones_like(lc), lc]
        if self.quadratic:
            cols.append(lc ** 2)
        cols.extend([d, rho])
        return np.stack(cols, axis=-1)

    def _idx_l(self):
        """Column index of the centered log sigma_total feature."""
        return 1

    # ------------------------------------------------------------------
    # Scale model
    # ------------------------------------------------------------------

    def _fit_scale_model(self, Z, log_r, tau):
        """Quantile regression of log|residual| on Z at level tau, guarded by baselines.

        Returns the coefficient vector achieving the lowest pinball loss among
        {Standard CP, CP-VS, Adam solution}, so the learned scale can never be worse than
        either baseline on the fitting objective.
        """
        n, p = Z.shape

        # Column scaling (intercept excluded) purely for conditioning of the optimizer.
        scale = np.ones(p)
        scale[1:] = np.maximum(Z[:, 1:].std(axis=0), 1e-6)
        Zs = Z / scale

        def loss(b):
            return _pinball(log_r - Zs @ b, tau)

        # --- Baseline 1: Standard CP, u = const. Optimal intercept is the tau-quantile.
        beta_std = np.zeros(p)
        beta_std[0] = np.quantile(log_r, tau)

        # --- Baseline 2: CP-VS, u = sigma_total, i.e. unit coefficient on (uncentered) l.
        i_l = self._idx_l()
        beta_cpvs = np.zeros(p)
        beta_cpvs[i_l] = scale[i_l]  # undo the column scaling -> coefficient 1 on lc
        beta_cpvs[0] = np.quantile(log_r - Zs @ beta_cpvs, tau)

        candidates = {"standard_cp": beta_std, "cpvs": beta_cpvs}
        losses = {k: loss(b) for k, b in candidates.items()}
        beta = candidates[min(losses, key=losses.get)].copy()

        # --- Full-batch Adam on the (convex) pinball objective.
        m = np.zeros(p)
        v = np.zeros(p)
        b1, b2, eps_adam = 0.9, 0.999, 1e-8
        penalty = np.ones(p) * self.ridge
        penalty[0] = 0.0
        best_beta, best_loss = beta.copy(), loss(beta)

        for t in range(1, self.n_iter + 1):
            e = log_r - Zs @ beta
            w = np.where(e < 0.0, tau - 1.0, tau)
            grad = -(Zs * w[:, None]).mean(axis=0) + penalty * beta
            m = b1 * m + (1 - b1) * grad
            v = b2 * v + (1 - b2) * grad ** 2
            mhat = m / (1 - b1 ** t)
            vhat = v / (1 - b2 ** t)
            step = self.lr * (1.0 - t / (self.n_iter + 1.0))  # linear decay
            beta = beta - step * mhat / (np.sqrt(vhat) + eps_adam)
            if t % 25 == 0 or t == self.n_iter:
                cur = loss(beta)
                if cur < best_loss:
                    best_loss, best_beta = cur, beta.copy()

        losses["fitted"] = best_loss
        self.fit_report_ = {
            "pinball_standard_cp": losses["standard_cp"],
            "pinball_cpvs": losses["cpvs"],
            "pinball_fitted": best_loss,
            "n_fit_rows": int(n),
        }
        # Undo column scaling so beta applies to the raw features.
        return best_beta / scale

    def scale(self, pred):
        """Evaluate the learned half-width scale u(x) for a MoGPrediction."""
        if self.beta is None:
            raise ValueError("CPDVSCalibrator must be calibrated before use.")
        Z = self._features(pred)
        log_u = Z @ self.beta
        # Clip before exponentiating: an unbounded log-scale would overflow on outliers.
        log_u = np.clip(log_u, -60.0, 60.0)
        return np.exp(log_u)

    # ------------------------------------------------------------------
    # Public split-conformal interface
    # ------------------------------------------------------------------

    def calibrate(self, cal_preds, cal_labels, alpha=None):
        """Fit the scale model and the conformal quantile on held-out calibration data.

        Parameters
        ----------
        cal_preds : MoGPrediction
            Mixture parameters on the calibration split, arrays [N, K, H, C].
        cal_labels : ndarray, shape [N, H, C]
            Observed targets on the calibration split.
        alpha : float, optional
            Overrides the miscoverage level set at construction.

        Returns
        -------
        self
        """
        if alpha is not None:
            self.alpha = alpha
        tau = 1.0 - self.alpha

        cal_labels = np.asarray(cal_labels, dtype=np.float64)
        if cal_labels.shape != cal_preds.shape:
            raise ValueError(f"Label shape {cal_labels.shape} != prediction shape {cal_preds.shape}")

        n = cal_labels.shape[0]
        n_fit = int(round(self.fit_frac * n))
        if n_fit < 2 or n - n_fit < 2:
            raise ValueError(f"Calibration split of {n} samples is too small for fit_frac={self.fit_frac}")

        # Time-ordered split: the leading block fits the scale, the trailing block
        # (closer to the test period) supplies the conformal quantile.
        fit_pred, fit_y = cal_preds.subset(slice(0, n_fit)), cal_labels[:n_fit]
        cq_pred, cq_y = cal_preds.subset(slice(n_fit, n)), cal_labels[n_fit:]

        self._l_mean = None  # recomputed from the fit split only
        Z = self._features(fit_pred)
        r = np.abs(fit_y - fit_pred.mu_bar)
        # Floor the residual relative to its own scale so log() stays finite without
        # letting the floor influence the upper tail the quantile regression targets.
        r_floor = max(1e-8, 1e-6 * float(np.mean(r)))
        log_r = np.log(np.maximum(r, r_floor))

        Zf = Z.reshape(-1, Z.shape[-1])
        log_rf = log_r.reshape(-1)
        if Zf.shape[0] > self.max_fit_rows:
            stride = int(np.ceil(Zf.shape[0] / self.max_fit_rows))
            Zf, log_rf = Zf[::stride], log_rf[::stride]

        self.beta = self._fit_scale_model(Zf, log_rf, tau)

        # Conformal step on the disjoint block, per (horizon step, channel).
        u = self.scale(cq_pred)
        scores = np.abs(cq_y - cq_pred.mu_bar) / np.maximum(u, EPS)
        n_cq = scores.shape[0]
        self.q_hat = np.quantile(scores, conformal_quantile_level(n_cq, self.alpha),
                                 axis=0, method="higher")
        self.fit_report_["n_conformal"] = int(n_cq)
        self.fit_report_["q_hat_mean"] = float(np.mean(self.q_hat))
        return self

    def predict_intervals(self, test_preds, alpha=None):
        """Return (lower, upper) prediction intervals, each [N, H, C].

        `alpha` may be passed for symmetry with `calibrate`, but it must match the level
        the calibrator was fit at -- the scale model is tuned to a specific tail.
        """
        if self.q_hat is None:
            raise ValueError("CPDVSCalibrator must be calibrated before predicting intervals.")
        if alpha is not None and not np.isclose(alpha, self.alpha):
            raise ValueError(
                f"Calibrated at alpha={self.alpha} but asked for alpha={alpha}; "
                f"re-run calibrate() at the target level.")
        half_width = self.q_hat * self.scale(test_preds)
        return test_preds.mu_bar - half_width, test_preds.mu_bar + half_width

    # ------------------------------------------------------------------
    # Online interface, matching AdaptiveCPVS's delayed-update protocol
    # ------------------------------------------------------------------

    def fit(self, cal_preds, cal_labels, window_size=1000):
        """Seed a rolling score window for online use (calls `calibrate` internally).

        The window is seeded from the *conformal* half only. Scoring the whole
        calibration block here would put the scale model's own training points into the
        window it is later quantiled over, biasing q_hat downward and voiding the
        split-conformal argument.
        """
        self.calibrate(cal_preds, cal_labels)
        n = np.asarray(cal_labels).shape[0]
        n_fit = int(round(self.fit_frac * n))
        cq_pred = cal_preds.subset(slice(n_fit, n))
        cq_y = np.asarray(cal_labels, dtype=np.float64)[n_fit:]

        scores = np.abs(cq_y - cq_pred.mu_bar) / np.maximum(self.scale(cq_pred), EPS)
        self.window_size = window_size
        self.scores_window = (scores[-window_size:].copy()
                              if scores.shape[0] > window_size else scores.copy())
        return self

    def interval_from_q(self, q, pred_t):
        """Build the interval at a fixed q without re-deriving it from the window."""
        width = q * self.scale(pred_t)
        return pred_t.mu_bar - width, pred_t.mu_bar + width

    def predict_one_step(self, pred_t):
        q = np.quantile(self.scores_window, conformal_quantile_level(self.scores_window.shape[0], self.alpha),
                        axis=0, method="higher")
        lower, upper = self.interval_from_q(q, pred_t)
        return lower, upper, q

    def update(self, pred_t, target_t):
        u = self.scale(pred_t)
        new_score = np.abs(np.asarray(target_t, dtype=np.float64) - pred_t.mu_bar) / np.maximum(u, EPS)
        self.scores_window = np.roll(self.scores_window, -1, axis=0)
        self.scores_window[-1] = new_score
