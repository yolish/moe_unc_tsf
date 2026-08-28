import numpy as np


class AdaptiveVarianceCalibrator:
    """Online variance-scaling CP with a learned epistemic ratio r.

    Time-series counterpart of the tabular AdaptiveVarianceCalibrator. Generalizes the
    aleatoric-scale calibrator by letting the epistemic term enter the calibrated
    variance with its own weight r rather than being fixed at 1:

        s     = max(0, ((y - y_hat)^2 - r * var_epi) / var_ale)
        width = sqrt(q^2 * var_ale + r * var_epi)

    r = 1 recovers AleatoricScaleCalibrator (sigma_i -> q*sigma_i, epistemic untouched),
    r = 0 recovers AleatoricOnlyCalibrator (epistemic dropped). Coverage does not depend
    on r -- the split-conformal step restores validity for any fixed r -- so r is chosen
    purely to minimize width.

    r is a structural choice, fit once on the validation split and then held constant;
    only q^2 is refreshed online from the sliding score window. The tabular version tunes
    r over random tune/calib permutations, which would destroy the temporal structure
    here, so the splits are contiguous blocks slid across the validation stream instead.
    Averaging the width curve over several origins keeps r* from latching onto whichever
    stretch of the series happens to land in a single tune fold.
    """

    def __init__(self, alpha=0.1, window_size=1000, r_grid=None, block_tuning=True,
                 tune_frac=0.5, min_block_n=200, n_repeats=20):
        self.alpha = alpha
        self.window_size = window_size
        self.r_grid = r_grid if r_grid is not None else \
            np.unique(np.concatenate(([0.0], np.geomspace(1e-2, 1e2, 25))))
        self.block_tuning = block_tuning
        self.tune_frac = tune_frac
        self.min_block_n = min_block_n
        self.n_repeats = n_repeats

        self.r = None
        self.scores_window = None
        self.tuning_widths_ = {}
        self.tuning_widths_std_ = {}
        self.n_tune_ = None

    def _calc_scores(self, preds, ale_var, epi_var, targets, r):
        residuals_sq = (targets - preds) ** 2
        var_ale = np.maximum(0, ale_var)
        var_epi = np.maximum(0, epi_var)

        scores = (residuals_sq - r * var_epi) / (var_ale + 1e-8)
        return np.maximum(0, scores)

    def _q_sq_from(self, scores):
        n = scores.shape[0]
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)

        q_squared = np.quantile(scores, q_level, axis=0, method='higher')
        return np.maximum(0, q_squared)

    def _mean_width(self, ale_var, epi_var, q_squared, r):
        variance = (q_squared * np.maximum(0, ale_var) + r * np.maximum(0, epi_var))
        return float(np.mean(np.sqrt(np.maximum(0, variance))))

    def _block_origins(self, n_total, n_tune):
        """Contiguous tune-block start indices, evenly spaced across the stream."""
        span = n_total - n_tune
        if span <= 0 or self.n_repeats <= 1:
            return [0]
        return sorted(set(np.linspace(0, span, self.n_repeats).round().astype(int).tolist()))

    def _tune_r(self, preds, ale_var, epi_var, targets):
        n_total = preds.shape[0]
        use_blocks = self.block_tuning and n_total >= self.min_block_n

        if use_blocks:
            n_tune = int(round(self.tune_frac * n_total))
            origins = self._block_origins(n_total, n_tune)
        else:
            n_tune = n_total
            origins = [0]

        width_curves = np.empty((len(origins), len(self.r_grid)))
        for b, start in enumerate(origins):
            sl = slice(start, start + n_tune)
            p, a, e, y = preds[sl], ale_var[sl], epi_var[sl], targets[sl]
            for j, r in enumerate(self.r_grid):
                q_sq = self._q_sq_from(self._calc_scores(p, a, e, y, r))
                width_curves[b, j] = self._mean_width(a, e, q_sq, r)

        avg_width_curve = width_curves.mean(axis=0)
        self.tuning_widths_ = {float(r): float(w) for r, w in zip(self.r_grid, avg_width_curve)}
        self.tuning_widths_std_ = {float(r): float(s)
                                   for r, s in zip(self.r_grid, width_curves.std(axis=0))}
        self.n_tune_ = n_tune

        best_j = 0
        for j in range(1, len(self.r_grid)):
            if avg_width_curve[j] < avg_width_curve[best_j] - 1e-12:
                best_j = j
        return float(self.r_grid[best_j])

    def fit(self, preds, ale_var, epi_var, targets):
        self.r = self._tune_r(preds, ale_var, epi_var, targets)

        scores = self._calc_scores(preds, ale_var, epi_var, targets, self.r)
        if scores.shape[0] > self.window_size:
            self.scores_window = scores[-self.window_size:]
        else:
            self.scores_window = scores

    def interval_from_q(self, q_squared, pred_t, ale_var_t, epi_var_t):
        """Build the interval for a given q^2 without re-deriving it from the window."""
        calibrated_variance = (np.maximum(0, q_squared) * np.maximum(0, ale_var_t)
                               + self.r * np.maximum(0, epi_var_t))
        width = np.sqrt(np.maximum(0, calibrated_variance))
        return pred_t - width, pred_t + width

    def predict_one_step(self, pred_t, ale_var_t, epi_var_t):
        if self.scores_window is None:
            raise ValueError("Calibrator must be initialized with fit() first!")

        q_squared = self._q_sq_from(self.scores_window)

        lower, upper = self.interval_from_q(q_squared, pred_t, ale_var_t, epi_var_t)
        return lower, upper, q_squared

    def update(self, pred_t, ale_var_t, epi_var_t, target_t):
        new_score = self._calc_scores(pred_t, ale_var_t, epi_var_t, target_t, self.r)

        if new_score.ndim == self.scores_window.ndim - 1:
            new_score = np.expand_dims(new_score, axis=0)

        if self.scores_window.shape[0] < self.window_size:
            self.scores_window = np.concatenate([self.scores_window, new_score], axis=0)
        else:
            self.scores_window = np.concatenate([self.scores_window[1:], new_score], axis=0)
