import numpy as np

from calibration.aci_core import ACIState, order_stat_quantile


class ACICQRCalibrator:
    """Conformalized Quantile Regression whose target level is driven by ACI.

    Score and interval are those of OnlineCQRQuantile -- the two-sided CQR conformity score
    of Romano et al. (2019), which widens or shrinks the model's own quantile head:

        s     = max(q_lo - y, y - q_hi)
        bounds = (q_lo - q, q_hi + q)

    What changes is where q is read off the rolling window: the level is a per-(horizon,
    channel) state variable updated by the ACI recursion of Gibbs & Candes (NeurIPS 2021)
    rather than pinned at 1 - alpha. See calibration/aci_core.py.

    gamma = 0 recovers OnlineCQRQuantile exactly.

    One class, two drivers. This serves both the plain CQR calibration and the rolling-
    retrain one, exactly as OnlineCQRQuantile does: retraining periodically re-fits the
    *model* between origins and changes the window size, neither of which touches the score,
    the quantile, or the update. There is no second calibrator to write.

    Note the score is signed: when the quantile head already over-covers, s is negative and
    so is q, which correctly *shrinks* the interval. Do not clamp q at 0 -- that would break
    parity with the base method at gamma = 0. `order_stat_quantile` floors its result at 0,
    so this calibrator indexes the sorted window itself rather than calling it.

    Harness contract: `interval_from_q` is *stateful* -- it records the level it served so
    the delayed feedback in `update` can score the right one. The driver must route every
    origin through `predict_one_step` or `interval_from_q`, never build the interval inline.
    """

    def __init__(self, alpha=0.1, gamma=0.01, window_size=1000):
        self.alpha = alpha
        self.window_size = window_size
        self.scores_window = None
        self.aci = ACIState(alpha=alpha, gamma=gamma)

    def _calc_scores(self, lower_preds, upper_preds, targets):
        return np.maximum(lower_preds - targets, targets - upper_preds)

    def fit(self, lower_preds, upper_preds, targets):
        scores = self._calc_scores(lower_preds, upper_preds, targets)

        if scores.shape[0] > self.window_size:
            self.scores_window = scores[-self.window_size:]
        else:
            self.scores_window = scores

        self.aci.reset(scores.shape[1:])

    def _current_q(self):
        """Per-cell order statistic, *without* the non-negativity floor.

        Same index rule as calibration.aci_core.order_stat_quantile -- the
        finite-sample-corrected level ceil((n+1)(1-alpha_t))/n capped at 1.0, then
        np.quantile's method='higher' rule ceil(level*(n-1)) -- but the CQR score is signed,
        so the floor that order_stat_quantile applies would clamp away legitimate negative
        q and diverge from OnlineCQRQuantile at gamma = 0.
        """
        n = self.scores_window.shape[0]

        alpha_t = np.clip(self.aci.alpha_t, 0.0, 1.0)
        q_level = np.minimum(np.ceil((n + 1) * (1.0 - alpha_t)) / n, 1.0)
        k = np.clip(np.ceil(q_level * (n - 1)).astype(int), 0, n - 1)

        ordered = np.sort(self.scores_window, axis=0)
        return np.take_along_axis(ordered, k[None], axis=0)[0]

    def interval_from_q(self, q, lower_t, upper_t):
        """Build the interval for a given q, recording it for the delayed ACI update."""
        self.aci.record(q)
        return lower_t - q, upper_t + q

    def predict_one_step(self, lower_t, upper_t):
        if self.scores_window is None:
            raise ValueError("Calibrator must be initialized with fit() first!")

        q = self._current_q()
        calibrated_lower, calibrated_upper = self.interval_from_q(q, lower_t, upper_t)
        return calibrated_lower, calibrated_upper, q

    def update(self, lower_t, upper_t, target_t):
        new_score = self._calc_scores(lower_t, upper_t, target_t)

        # Score against the level actually served at this origin *before* the observation
        # joins the window -- otherwise it would influence the q it is being scored against.
        self.aci.step(new_score, fallback_q=self._current_q())

        if self.scores_window is not None:
            if new_score.ndim == self.scores_window.ndim - 1:
                new_score = np.expand_dims(new_score, axis=0)
            elif new_score.ndim == 0 and self.scores_window.ndim == 1:
                new_score = np.expand_dims(new_score, axis=0)

        if self.scores_window.shape[0] < self.window_size:
            self.scores_window = np.concatenate([self.scores_window, new_score], axis=0)
        else:
            self.scores_window = np.concatenate([self.scores_window[1:], new_score], axis=0)
