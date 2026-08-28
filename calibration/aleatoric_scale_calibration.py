import numpy as np


class AleatoricScaleCalibrator:
    """Online variance-scaling CP that calibrates only the aleatoric component.

    Time-series counterpart of the tabular AleatoricScaleCalibrator. The MoG predictive
    variance splits as var_total = var_ale + var_epi (law of total variance); replacing
    each component's sigma_i by q*sigma_i rescales the aleatoric term alone:

        Var(y|x) = q^2 * var_ale(x) + var_epi(x)

    so the conformity score subtracts the (uncalibrated) epistemic floor before dividing
    by the aleatoric variance, and the interval adds that floor back in:

        s     = max(0, ((y - y_hat)^2 - var_epi) / var_ale)
        width = sqrt(q^2 * var_ale + var_epi)

    Clamping at 0 does not affect coverage: since q^2 >= 0, the event {s <= q^2} is the
    same before and after clamping, so the usual conformal guarantee is untouched.

    Unlike the tabular version this keeps a sliding window of scores and refreshes q^2
    online, so the calibration tracks drift over the test stream.
    """

    def __init__(self, alpha=0.1, window_size=1000):
        self.alpha = alpha
        self.window_size = window_size
        self.scores_window = None

    def _calc_scores(self, preds, ale_var, epi_var, targets):
        residuals_sq = (targets - preds) ** 2
        var_ale = np.maximum(0, ale_var)
        var_epi = np.maximum(0, epi_var)

        scores = (residuals_sq - var_epi) / (var_ale + 1e-8)
        return np.maximum(0, scores)

    def fit(self, preds, ale_var, epi_var, targets):
        scores = self._calc_scores(preds, ale_var, epi_var, targets)

        if scores.shape[0] > self.window_size:
            self.scores_window = scores[-self.window_size:]
        else:
            self.scores_window = scores

    def interval_from_q(self, q_squared, pred_t, ale_var_t, epi_var_t):
        """Build the interval for a given q^2 without re-deriving it from the window."""
        calibrated_variance = (np.maximum(0, q_squared) * np.maximum(0, ale_var_t)
                               + np.maximum(0, epi_var_t))
        width = np.sqrt(np.maximum(0, calibrated_variance))
        return pred_t - width, pred_t + width

    def predict_one_step(self, pred_t, ale_var_t, epi_var_t):
        if self.scores_window is None:
            raise ValueError("Calibrator must be initialized with fit() first!")

        n = self.scores_window.shape[0]
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)

        q_squared = np.quantile(self.scores_window, q_level, axis=0, method='higher')
        q_squared = np.maximum(0, q_squared)

        lower, upper = self.interval_from_q(q_squared, pred_t, ale_var_t, epi_var_t)
        return lower, upper, q_squared

    def update(self, pred_t, ale_var_t, epi_var_t, target_t):
        new_score = self._calc_scores(pred_t, ale_var_t, epi_var_t, target_t)

        if new_score.ndim == self.scores_window.ndim - 1:
            new_score = np.expand_dims(new_score, axis=0)

        if self.scores_window.shape[0] < self.window_size:
            self.scores_window = np.concatenate([self.scores_window, new_score], axis=0)
        else:
            self.scores_window = np.concatenate([self.scores_window[1:], new_score], axis=0)
