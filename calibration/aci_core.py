"""Shared Adaptive Conformal Inference (Gibbs & Candes, NeurIPS 2021) machinery.

This module holds the two pieces every ACI calibrator in this package needs and nothing
else: the level recursion itself (`ACIState`) and the per-cell empirical quantile that the
recursion forces (`order_stat_quantile`). It is not a calibration method -- the score and
the interval always stay in the per-method file.

Why a separate module. The recursion is short but its correctness rests on an invariant
that is easy to break silently: the error at time t must be scored against the level that
was actually *served* at t, not against whatever the window says when the observation
finally arrives. Because this harness feeds observations back `pred_len` origins late, the
served levels have to be queued. Copy-pasting that queue into five calibrators is how one
of them ends up popping the wrong entry and adapting against a level it never issued, with
no error and only a slightly-wrong coverage number to show for it.

Deliberate duplication: `ACIAleatoricScaleCalibrator` predates this module and keeps its
own inlined copy of the same recursion. Its numbers are already published in
docs/calibration_results_tsf.md, so it is left alone rather than re-derived through here.
If you change the semantics below, that file does *not* follow -- check it by hand.
"""

import collections

import numpy as np


def order_stat_quantile(scores_window, alpha_t):
    """Per-cell empirical quantile of `scores_window` at the per-cell level 1 - alpha_t.

    np.quantile takes a scalar level, so once alpha_t varies per (horizon, channel) the
    quantile has to be done by hand: sort along the window axis and index. The two rules
    below reproduce the fixed-alpha calibrators exactly -- the finite-sample-corrected level
    ceil((n+1)(1-alpha))/n capped at 1.0, then np.quantile's method='higher' index rule
    ceil(level * (n-1)). Verified bit-identical to
    np.quantile(w, min(ceil((n+1)(1-a))/n, 1.0), axis=0, interpolation='higher')
    for n in {7, 100, 1000} and alpha in {0.1, 0.05, 0.001, 0}.

    Capping the level at 1.0 is the clip-to-window-max rule: once alpha_t is too small for n
    scores to express, the interval stops widening rather than going unbounded.

    Args:
        scores_window: [W, ...] rolling window of conformity scores, window axis first.
        alpha_t: working level array broadcastable to scores_window.shape[1:].

    Returns:
        Array of shape scores_window.shape[1:]; never negative.
    """
    n = scores_window.shape[0]

    alpha_t = np.clip(alpha_t, 0.0, 1.0)
    q_level = np.minimum(np.ceil((n + 1) * (1.0 - alpha_t)) / n, 1.0)
    k = np.clip(np.ceil(q_level * (n - 1)).astype(int), 0, n - 1)

    # A full sort beats bracketing the needed order statistics with np.partition here: the
    # scores are float32, where numpy's sort takes the AVX-512 vectorized path and
    # np.partition does not (measured on [1000, 96, 862]: sort 0.90s, partition-then-sort-
    # the-band 1.59s). Do not "optimize" this into a partition without re-measuring.
    ordered = np.sort(scores_window, axis=0)
    q = np.take_along_axis(ordered, k[None], axis=0)[0]

    return np.maximum(0.0, q)


class ACIState:
    """The Gibbs & Candes level recursion, decoupled from any score or interval.

    Held by composition rather than inheritance: each calibrator keeps its own score and
    interval and hands this object (a) the cell shape to track, (b) the level it served at
    each origin, (c) the score that comes back pred_len origins later.

        err_t       = 1{ score_t > q_served_t }
        alpha_{t+1} = clip(alpha_t + gamma * (alpha - err_t), 0, 1)

    A miss pushes alpha_t down, which widens future intervals; a cover pushes it up, which
    narrows them. `alpha` (the target) is never mutated -- only `alpha_t`, per cell, so each
    (horizon, channel) tracks its own miscoverage. gamma = 0 leaves alpha_t at alpha forever
    and therefore reproduces the fixed-alpha base method exactly.

    On the [0, 1] clip: in the paper alpha_t ranges over all of R, and a cell that runs past
    what the window can express serves an unbounded interval, forcing err_t = 0 and pulling
    itself back up. Clipped, such a cell parks at alpha_t = 0, keeps missing, and has no
    restoring force (integrator windup). The clip is kept so widths stay finite and
    comparable across the result files; p05 of alpha_t near 0 is the diagnostic that a run
    hit this regime. MoECP is the exception -- its localized quantile can return +inf, which
    restores the paper's own restoring force. See that calibrator's docstring.
    """

    def __init__(self, alpha=0.1, gamma=0.01):
        self.alpha = alpha
        self.gamma = gamma

        self.alpha_t = None
        self.q_pending = collections.deque()

        self.alpha_t_history = []
        self.err_rate_ = float('nan')
        self._err_sum = 0.0
        self._err_count = 0

    def reset(self, cell_shape):
        """Start a fresh stream over cells of shape `cell_shape` (e.g. (H, C))."""
        self.alpha_t = np.full(cell_shape, float(self.alpha))
        self.q_pending.clear()

        self.alpha_t_history = []
        self.err_rate_ = float('nan')
        self._err_sum = 0.0
        self._err_count = 0

    def record(self, q):
        """Record the level served at this origin.

        Must be called exactly once per origin, on every origin -- including origins whose q
        was reused from a cache. Skipping it on any origin shifts the queue permanently and
        every subsequent update scores against a level it never issued.
        """
        self.q_pending.append(q)

    def step(self, new_score, fallback_q=None):
        """Apply the recursion for one delayed observation.

        A non-finite served level scores as covered (err_t = 0), which raises alpha_t and so
        lowers the next target level. For MoECP, whose localized quantile can legitimately
        return +inf, that is the intended restoring force rather than a case to mask out --
        see ACIMoECPCalibrator's docstring.

        Args:
            new_score: conformity score of the observation now arriving.
            fallback_q: level to score against if the queue is empty (warm-up only).
        """
        q_used = self.q_pending.popleft() if self.q_pending else fallback_q
        err_t = (new_score > q_used).astype(float)

        self.alpha_t = np.clip(self.alpha_t + self.gamma * (self.alpha - err_t), 0.0, 1.0)
        self.alpha_t_history.append(float(np.mean(self.alpha_t)))

        self._err_sum += float(np.mean(err_t))
        self._err_count += 1
        self.err_rate_ = self._err_sum / self._err_count

        return err_t

    @property
    def level(self):
        """Per-cell target coverage level 1 - alpha_t."""
        return 1.0 - self.alpha_t

    def result_suffix(self):
        """Trailing key/value fields for a result line.

        Key names match those written by calibrate_aci_aleatoric_scale so the collector's
        kvpairs() reads every ACI method's line identically.
        """
        return (f", gamma: {self.gamma}"
                f", alpha_t_mean: {np.mean(self.alpha_t_history):.4f}"
                f", alpha_t_final: {np.mean(self.alpha_t):.4f}"
                f", alpha_t_p05: {np.percentile(self.alpha_t, 5):.4f}"
                f", alpha_t_p95: {np.percentile(self.alpha_t, 95):.4f}"
                f", err_rate: {self.err_rate_:.4f}")

    def summary_line(self):
        """One-line console diagnostic, mirroring the existing ACI methods' print."""
        return (f"Final alpha_t = {np.mean(self.alpha_t):.4f} mean, "
                f"[{np.percentile(self.alpha_t, 5):.4f}, "
                f"{np.percentile(self.alpha_t, 95):.4f}] p05-p95 "
                f"(target {self.alpha}), realized miscoverage = {self.err_rate_:.4f}")
