import numpy as np


class AdaptiveWindowAleatoricScaleCalibrator:
    """Aleatoric-scale CP whose sliding-window size is selected from the data.

    The score and the interval are exactly those of AleatoricScaleCalibrator:

        s     = max(0, ((y - y_hat)^2 - var_epi) / var_ale)
        width = sqrt(q^2 * var_ale + var_epi)

    What changes is where q^2 comes from. A fixed window hardcodes an assumption about
    how fast the residual scale drifts: too short and q^2 is a noisy order statistic of a
    handful of points, too long and the calibration lags a regime change. The right
    length is a property of the series, not a constant -- ETT gives thousands of
    validation points while national_illness gives a few dozen, where a window of 1000
    never fills and the "sliding" window never slides at all.

    So the window size is chosen by replaying the deployment protocol on the tail of the
    validation split. For each candidate W the same delayed online loop the experiment
    harness runs (q refreshed from the rolling window, each observation fed back pred_len
    steps late) is simulated over a held-out validation region, and W is scored by the
    interval score (Winkler)

        IS = (u - l) + (2/alpha)(l - y)1{y<l} + (2/alpha)(y - u)1{y>u}

    the proper scoring rule for a central (1-alpha) interval. Width alone would be the
    wrong objective: it is minimized by whichever W happens to produce the smallest
    quantile, undercoverage included. The interval score makes a candidate pay for
    narrow-but-missing intervals through the (2/alpha) penalty, so it trades the two off
    at the level the calibrator is actually targeting.

    Candidates larger than the available history are not discarded -- they degrade to a
    growing window that never rolls, which is a genuine strategy and is scored as such.

    Coverage is unaffected by the choice: any fixed W gives a valid conformal quantile,
    so W is a width/adaptivity knob, not a validity one.
    """

    DEFAULT_GRID = (20, 30, 50, 75, 100, 150, 250, 400, 600, 1000, 1500, 2000)

    def __init__(self, alpha=0.1, pred_len=1, window_grid=None, min_window=20,
                 eval_frac=0.5, min_eval=30):
        self.alpha = alpha
        self.pred_len = pred_len
        self.window_grid = tuple(window_grid) if window_grid is not None else self.DEFAULT_GRID
        self.min_window = min_window
        self.eval_frac = eval_frac
        self.min_eval = min_eval

        self.window_size = None
        self.scores_window = None
        self.tuning_scores_ = {}
        self.n_eval_ = None

    # ---- shared score / interval algebra (identical to AleatoricScaleCalibrator) ----

    def _calc_scores(self, preds, ale_var, epi_var, targets):
        residuals_sq = (targets - preds) ** 2
        scores = (residuals_sq - np.maximum(0, epi_var)) / (np.maximum(0, ale_var) + 1e-8)
        return np.maximum(0, scores)

    def _q_sq_from(self, scores):
        n = scores.shape[0]
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        return np.maximum(0, np.quantile(scores, q_level, axis=0, method='higher'))

    def interval_from_q(self, q_squared, pred_t, ale_var_t, epi_var_t):
        """Build the interval for a given q^2 without re-deriving it from the window."""
        calibrated_variance = (np.maximum(0, q_squared) * np.maximum(0, ale_var_t)
                               + np.maximum(0, epi_var_t))
        width = np.sqrt(np.maximum(0, calibrated_variance))
        return pred_t - width, pred_t + width

    # ---- window-size selection ----

    def _interval_score(self, lower, upper, y):
        return ((upper - lower)
                + (2.0 / self.alpha) * np.maximum(0, lower - y)
                + (2.0 / self.alpha) * np.maximum(0, y - upper))

    def _replay(self, scores, preds, ale_var, epi_var, targets, start, window):
        """Simulate the harness loop from `start` onward with a given window size.

        Mirrors _run_separated_calibration: the window is seeded with the `window` scores
        preceding `start`, q is refreshed each step from that window, and the observation
        at t - pred_len is appended once it would have been observable.
        """
        win = scores[max(0, start - window):start]
        if win.shape[0] == 0:
            return np.inf

        total, n_scored = 0.0, 0
        for t in range(start, scores.shape[0]):
            q_sq = self._q_sq_from(win)
            lo, hi = self.interval_from_q(q_sq, preds[t], ale_var[t], epi_var[t])
            total += float(np.mean(self._interval_score(lo, hi, targets[t])))
            n_scored += 1

            t_update = t - self.pred_len
            if t_update >= start:
                new_score = scores[t_update:t_update + 1]
                win = np.concatenate([win, new_score], axis=0)
                if win.shape[0] > window:
                    win = win[-window:]

        return total / max(1, n_scored)

    def _tune_window(self, scores, preds, ale_var, epi_var, targets):
        n = scores.shape[0]
        n_eval = max(self.min_eval, int(round(self.eval_frac * n)))
        n_eval = min(n_eval, n - self.min_window)
        if n_eval <= 0:
            # Too little validation data to hold anything out; fall back to using it all.
            self.n_eval_ = 0
            return min(max(n, self.min_window), max(self.window_grid))

        start = n - n_eval
        self.n_eval_ = n_eval

        candidates = sorted({w for w in self.window_grid if w >= self.min_window})
        for w in candidates:
            self.tuning_scores_[int(w)] = self._replay(
                scores, preds, ale_var, epi_var, targets, start, w)

        best = min(self.tuning_scores_, key=lambda w: (self.tuning_scores_[w], w))
        return int(best)

    # ---- calibrator interface ----

    def fit(self, preds, ale_var, epi_var, targets):
        scores = self._calc_scores(preds, ale_var, epi_var, targets)
        self.window_size = self._tune_window(scores, preds, ale_var, epi_var, targets)

        if scores.shape[0] > self.window_size:
            self.scores_window = scores[-self.window_size:]
        else:
            self.scores_window = scores

    def predict_one_step(self, pred_t, ale_var_t, epi_var_t):
        if self.scores_window is None:
            raise ValueError("Calibrator must be initialized with fit() first!")

        q_squared = self._q_sq_from(self.scores_window)
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
