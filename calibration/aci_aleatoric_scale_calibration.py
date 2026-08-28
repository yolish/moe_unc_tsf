import collections

import numpy as np


class ACIAleatoricScaleCalibrator:
    """Aleatoric-scale CP whose target level is driven by Adaptive Conformal Inference.

    Score and interval are those of AleatoricScaleCalibrator -- the MoG predictive variance
    splits as var_total = var_ale + var_epi, and only the aleatoric term is rescaled:

        s     = max(0, ((y - y_hat)^2 - var_epi) / var_ale)
        width = sqrt(q^2 * var_ale + var_epi)

    What changes is where q^2 is read off the rolling window. AleatoricScaleCalibrator pins
    the level at 1 - alpha and lets adaptivity come entirely from which scores sit in the
    window; under drift that leaves nothing to push a systematically over- or under-covering
    quantile back. Here the level itself is a state variable, updated by the ACI recursion of
    Gibbs & Candes (NeurIPS 2021):

        alpha_{t+1} = alpha_t + gamma * (alpha - err_t),   err_t = 1{ s_t > q_t^2 }

    so realized coverage is steered toward the nominal 1 - alpha over the test stream. alpha
    (the target) is never mutated; alpha_t (the working level) is, per (horizon, channel)
    cell, so each horizon and channel tracks its own miscoverage.

    gamma = 0 recovers AleatoricScaleCalibrator exactly.

    alpha_t is clipped to [0, 1], and the quantile level is capped at 1.0
    ---------------------------------------------------------------------
    In Gibbs & Candes, alpha_t ranges over all of R: once it runs past what an n-score window
    can express (alpha_t < 1/(n+1)) the exact update serves C_t = R, which forces err_t = 0 and
    pulls alpha_t back up. That unboundedness is what carries their coverage bound -- clipped,
    a cell that cannot be covered by the window maximum instead parks at alpha_t = 0, keeps
    missing, and has no restoring force acting on it (integrator windup), which measurably
    undercovers relative to the exact recursion.

    The clip is kept anyway: the interval stays finite on every cell, which is what makes width
    a single comparable number against the sibling calibrators in the TSF result files, rather
    than requiring every comparison to also carry an unbounded-interval fraction. That is a
    real accuracy-vs-comparability trade, made deliberately -- see docs/calibration_results_tsf.md
    for the gamma choice this trade was evaluated under.

    Note on the harness contract: unlike its siblings, `interval_from_q` here is *stateful* --
    it records the level it served so the delayed feedback in `update` can score the right
    one. The driver (`_run_separated_calibration`) calls exactly one of `predict_one_step` /
    `interval_from_q` per origin, which is what keeps the record aligned with the stream.
    """

    def __init__(self, alpha=0.1, gamma=0.01, window_size=1000):
        self.alpha = alpha
        self.gamma = gamma
        self.window_size = window_size
        self.scores_window = None

        self.alpha_t = None
        self.q_pending = collections.deque()

        self.alpha_t_history = []
        self.err_rate_ = float('nan')
        self._err_sum = 0.0
        self._err_count = 0

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

        self.alpha_t = np.full(scores.shape[1:], float(self.alpha))
        self.q_pending.clear()

    def _current_q(self):
        """Per-cell empirical quantile of the window at the per-cell level 1 - alpha_t.

        np.quantile takes a scalar level, so with alpha_t varying per (H, C) the quantile is
        done by hand: sort along the window axis and index. The two lines below reproduce
        AleatoricScaleCalibrator exactly -- the finite-sample-corrected level
        ceil((n+1)(1-alpha))/n capped at 1.0, then np.quantile's method='higher' index rule
        ceil(level * (n-1)). Capping the level at 1.0 is the clip-to-window-max rule: once
        alpha_t is too small for n scores to express, the interval stops widening.
        """
        n = self.scores_window.shape[0]

        alpha_t = np.clip(self.alpha_t, 0.0, 1.0)
        q_level = np.minimum(np.ceil((n + 1) * (1.0 - alpha_t)) / n, 1.0)
        k = np.clip(np.ceil(q_level * (n - 1)).astype(int), 0, n - 1)

        # A full sort beats bracketing the needed order statistics with np.partition here:
        # the scores are float32, where numpy's sort takes the AVX-512 vectorized path and
        # np.partition does not (measured on [1000, 96, 862]: sort 0.90s, partition-then-
        # sort-the-band 1.59s). Do not "optimize" this into a partition without re-measuring.
        ordered = np.sort(self.scores_window, axis=0)
        q_squared = np.take_along_axis(ordered, k[None], axis=0)[0]

        return np.maximum(0.0, q_squared)

    def interval_from_q(self, q_squared, pred_t, ale_var_t, epi_var_t):
        """Build the interval for a given q^2, recording it for the delayed ACI update."""
        self.q_pending.append(q_squared)

        calibrated_variance = (np.maximum(0, q_squared) * np.maximum(0, ale_var_t)
                               + np.maximum(0, epi_var_t))
        width = np.sqrt(np.maximum(0, calibrated_variance))
        return pred_t - width, pred_t + width

    def predict_one_step(self, pred_t, ale_var_t, epi_var_t):
        if self.scores_window is None:
            raise ValueError("Calibrator must be initialized with fit() first!")

        q_squared = self._current_q()
        lower, upper = self.interval_from_q(q_squared, pred_t, ale_var_t, epi_var_t)
        return lower, upper, q_squared

    def update(self, pred_t, ale_var_t, epi_var_t, target_t):
        new_score = self._calc_scores(pred_t, ale_var_t, epi_var_t, target_t)

        # The harness feeds each observation back pred_len origins late, so the error has to
        # be scored against the q^2 actually served at that origin -- the oldest pending one.
        q_used = self.q_pending.popleft() if self.q_pending else self._current_q()
        err_t = (new_score > q_used).astype(float)

        self.alpha_t = np.clip(self.alpha_t + self.gamma * (self.alpha - err_t), 0.0, 1.0)
        self.alpha_t_history.append(float(np.mean(self.alpha_t)))

        self._err_sum += float(np.mean(err_t))
        self._err_count += 1
        self.err_rate_ = self._err_sum / self._err_count

        # Only now does the scored observation join the window that will produce future q^2.
        if new_score.ndim == self.scores_window.ndim - 1:
            new_score = np.expand_dims(new_score, axis=0)

        if self.scores_window.shape[0] < self.window_size:
            self.scores_window = np.concatenate([self.scores_window, new_score], axis=0)
        else:
            self.scores_window = np.concatenate([self.scores_window[1:], new_score], axis=0)
