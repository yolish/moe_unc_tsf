import numpy as np


class AleatoricOnlyCalibrator:
    """Online variance-scaling CP driven by the aleatoric variance alone.

    The conformity score is the squared residual over the aleatoric variance, so the
    calibrated quantile q^2 acts as a multiplicative correction on that variance:
    width = sqrt(q^2 * var_ale). The epistemic component is deliberately ignored.
    """

    def __init__(self, alpha=0.1, window_size=1000):
        self.alpha = alpha
        self.window_size = window_size
        self.scores_window = None

    def fit(self, preds, ale_var, targets):
        residuals_sq = (targets - preds) ** 2
        scores = residuals_sq / (ale_var + 1e-8)

        if scores.shape[0] > self.window_size:
            self.scores_window = scores[-self.window_size:]
        else:
            self.scores_window = scores

    def interval_from_q(self, q_squared, pred_t, ale_var_t):
        """Build the interval for a given q^2 without re-deriving it from the window."""
        calibrated_variance = np.maximum(0, q_squared) * ale_var_t
        width = np.sqrt(np.maximum(0, calibrated_variance))
        return pred_t - width, pred_t + width

    def predict_one_step(self, pred_t, ale_var_t):
        if self.scores_window is None:
            raise ValueError("Calibrator must be initialized with fit() first!")

        n = self.scores_window.shape[0]
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)

        q_squared = np.quantile(self.scores_window, q_level, axis=0, method='higher')
        q_squared = np.maximum(0, q_squared)

        lower, upper = self.interval_from_q(q_squared, pred_t, ale_var_t)
        return lower, upper, q_squared

    def update(self, pred_t, ale_var_t, target_t):
        res_sq = (target_t - pred_t) ** 2
        new_score = res_sq / (ale_var_t + 1e-8)

        if new_score.ndim == self.scores_window.ndim - 1:
            new_score = np.expand_dims(new_score, axis=0)

        if self.scores_window.shape[0] < self.window_size:
            self.scores_window = np.concatenate([self.scores_window, new_score], axis=0)
        else:
            self.scores_window = np.concatenate([self.scores_window[1:], new_score], axis=0)
