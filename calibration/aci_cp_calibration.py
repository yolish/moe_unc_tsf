import numpy as np

from calibration.aci_core import ACIState, order_stat_quantile


class ACICPCalibrator:
    """Standard split CP whose target level is driven by Adaptive Conformal Inference.

    Score and interval are those of AdaptiveCP -- the plain absolute residual, with no
    variance scaling of any kind:

        s     = |y - y_hat|
        width = q

    What changes is where q is read off the rolling window. AdaptiveCP pins the level at
    1 - alpha and lets adaptivity come entirely from which residuals sit in the window;
    under drift that leaves nothing to push a systematically over- or under-covering
    quantile back. Here the level is a state variable updated by the ACI recursion of
    Gibbs & Candes (NeurIPS 2021), per (horizon, channel) cell. See calibration/aci_core.py.

    gamma = 0 recovers AdaptiveCP exactly.

    Harness contract: `interval_from_q` is *stateful* -- it records the level it served so
    the delayed feedback in `update` can score the right one. The driver must route every
    origin through `predict_one_step` or `interval_from_q`, never build the interval inline,
    or the record falls out of step with the stream.
    """

    def __init__(self, alpha=0.1, gamma=0.01, window_size=1000):
        self.alpha = alpha
        self.window_size = window_size
        self.scores_window = None
        self.aci = ACIState(alpha=alpha, gamma=gamma)

    def _calc_scores(self, preds, targets):
        return np.abs(targets - preds)

    def fit(self, preds, targets):
        scores = self._calc_scores(preds, targets)

        if scores.shape[0] > self.window_size:
            self.scores_window = scores[-self.window_size:]
        else:
            self.scores_window = scores

        self.aci.reset(scores.shape[1:])

    def _current_q(self):
        return order_stat_quantile(self.scores_window, self.aci.alpha_t)

    def interval_from_q(self, q, pred_t):
        """Build the interval for a given q, recording it for the delayed ACI update."""
        self.aci.record(q)
        return pred_t - q, pred_t + q

    def predict_one_step(self, pred_t):
        if self.scores_window is None:
            raise ValueError("Calibrator must be initialized with fit() first!")

        q = self._current_q()
        lower, upper = self.interval_from_q(q, pred_t)
        return lower, upper, q

    def update(self, pred_t, target_t):
        new_score = self._calc_scores(pred_t, target_t)

        # Score against the level actually served at this origin *before* the observation
        # joins the window -- otherwise it would influence the q it is being scored against.
        self.aci.step(new_score, fallback_q=self._current_q())

        if new_score.ndim == self.scores_window.ndim - 1:
            new_score = np.expand_dims(new_score, axis=0)

        if self.scores_window.shape[0] < self.window_size:
            self.scores_window = np.concatenate([self.scores_window, new_score], axis=0)
        else:
            self.scores_window = np.concatenate([self.scores_window[1:], new_score], axis=0)
