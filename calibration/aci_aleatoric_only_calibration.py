import numpy as np

from calibration.aci_core import ACIState, order_stat_quantile


class ACIAleatoricOnlyCalibrator:
    """Aleatoric-only variance-scaling CP whose target level is driven by ACI.

    Score and interval are those of AleatoricOnlyCalibrator -- the squared residual over the
    aleatoric variance, so the calibrated q^2 is a multiplicative correction on that
    variance and the epistemic component is deliberately ignored:

        s     = (y - y_hat)^2 / var_ale
        width = sqrt(q^2 * var_ale)

    What changes is where q^2 is read off the rolling window: the level is a per-(horizon,
    channel) state variable updated by the ACI recursion of Gibbs & Candes (NeurIPS 2021)
    rather than pinned at 1 - alpha. See calibration/aci_core.py.

    gamma = 0 recovers AleatoricOnlyCalibrator exactly.

    This is the one member of the family that needs no new driver: the base method already
    exposes a real `interval_from_q` seam and runs through `_run_separated_calibration`,
    whose cache-hit branch calls it -- so the served-level record stays aligned with the
    stream for free. Its sibling drivers (calibrate_cp, calibrate_cpvs, calibrate_cqr) build
    the interval inline on cache hits instead, which is why they need ACI-specific loops.
    """

    def __init__(self, alpha=0.1, gamma=0.01, window_size=1000):
        self.alpha = alpha
        self.window_size = window_size
        self.scores_window = None
        self.aci = ACIState(alpha=alpha, gamma=gamma)

    def _calc_scores(self, preds, ale_var, targets):
        residuals_sq = (targets - preds) ** 2
        return residuals_sq / (ale_var + 1e-8)

    def fit(self, preds, ale_var, targets):
        scores = self._calc_scores(preds, ale_var, targets)

        if scores.shape[0] > self.window_size:
            self.scores_window = scores[-self.window_size:]
        else:
            self.scores_window = scores

        self.aci.reset(scores.shape[1:])

    def _current_q(self):
        return order_stat_quantile(self.scores_window, self.aci.alpha_t)

    def interval_from_q(self, q_squared, pred_t, ale_var_t):
        """Build the interval for a given q^2, recording it for the delayed ACI update."""
        self.aci.record(q_squared)

        calibrated_variance = np.maximum(0, q_squared) * ale_var_t
        width = np.sqrt(np.maximum(0, calibrated_variance))
        return pred_t - width, pred_t + width

    def predict_one_step(self, pred_t, ale_var_t):
        if self.scores_window is None:
            raise ValueError("Calibrator must be initialized with fit() first!")

        q_squared = self._current_q()
        lower, upper = self.interval_from_q(q_squared, pred_t, ale_var_t)
        return lower, upper, q_squared

    def update(self, pred_t, ale_var_t, target_t):
        new_score = self._calc_scores(pred_t, ale_var_t, target_t)

        # Score against the level actually served at this origin *before* the observation
        # joins the window -- otherwise it would influence the q it is being scored against.
        self.aci.step(new_score, fallback_q=self._current_q())

        if new_score.ndim == self.scores_window.ndim - 1:
            new_score = np.expand_dims(new_score, axis=0)

        if self.scores_window.shape[0] < self.window_size:
            self.scores_window = np.concatenate([self.scores_window, new_score], axis=0)
        else:
            self.scores_window = np.concatenate([self.scores_window[1:], new_score], axis=0)
