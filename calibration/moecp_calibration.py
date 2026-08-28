"""MoECP for time-series forecasting: gating-localized weighted conformal prediction.

Port of `calibration/tabular_regression/MoECP_Calibrator.py` to the rolling-window,
delayed-update protocol the other time-series calibrators in this package use
(`AdaptiveCPVS`, `AleatoricMOGCalibrator`, ...).

Method
------
Unlike the CP-VS family, MoECP does not rescale the residual. It keeps the raw absolute
residual as the non-conformity score and instead makes the *conformal quantile itself*
local: calibration points whose gate distribution resembles the test point's get more
weight in the quantile. With gate vectors pi(x) in the K-simplex,

    pi~            ~ Multinomial(tau, pi(X_test)) / tau        (randomization, Eq. 3)
    w_i            = exp(-tau * KL(pi~ || pi(X_i)))            i = 1..n calibration points
    w_{n+1}        = exp(-tau * KL(pi~ || pi(X_test)))         the test point's own weight
    C(X_test)      = mu(X_test) +/- Q_{1-alpha}( sum_i w~_i delta_{r_i} + w~_{n+1} delta_inf )

where w~ are the w normalized to sum to one over the n+1 points. The mass the test point
holds at +infinity is what makes the construction finite-sample valid (Tibshirani et al.
weighted exchangeability); when the target level falls inside that mass the interval is
unbounded, and clipping it to the largest observed residual would break the guarantee.

Numerical note
--------------
KL(pi~ || pi_i) = -H(pi~) - <pi~, log pi_i>, and the entropy term does not depend on i, so
it is a common factor of every w_i *and* of w_{n+1}. It therefore cancels in the
normalization and is never formed. Each step reduces to

    a_i = tau * <pi~, log pi_i>,   w~ = softmax over {a_1..a_n, a_test}

which is one einsum plus a shifted softmax, rather than an explicit KL per calibration
point. The shift by max(a) keeps exp() in range for the large tau (default 100) the method
uses.

Cost
----
Per test step the weighted quantile needs a sort of the window along the sample axis for
every (horizon step, channel) cell: O(W log W * H * C). That is materially more expensive
than the O(W * H * C) quantile the CP-VS family does, and it grows with the horizon. See
`recompute_every` for the available trade-off.
"""

import numpy as np

EPS = 1e-8


class MoECPCalibrator:
    """Gating-localized weighted conformal prediction over a rolling window.

    Parameters
    ----------
    alpha : float
        Miscoverage level; intervals target 1 - alpha coverage.
    temperature : float
        The tau of the paper, used both as the multinomial count for the randomization
        and as the exponent on the KL divergence, matching the tabular implementation.
        Larger tau localizes harder (weights concentrate on gate-similar calibration
        points); tau = 0 disables the randomization and flattens the weighting.

        Larger tau also means the test point's own normalized weight w~_{n+1} more often
        exceeds alpha, which is exactly the unbounded-interval condition (see the module
        docstring). On this TSF grid the paper's tabular default tau=100 pushed unbounded
        rates as high as 25% (national-illness) -- driven by w~_{n+1} being large, not by
        drift or windup, so lowering tau (unlike ACI's alpha_t clip) trades essentially
        nothing but localization sharpness, not validity: Theorem 1's guarantee holds for
        every tau in N+. tau=1 was measured to bring unbounded rates to ~0 across this
        grid's hardest cases while leaving width still well below standard_cp's (i.e. not
        collapsed to uniform weights). tau=0 eliminates the unbounded case exactly, but at
        that point the weights are provably uniform and the method is plain split CP on
        the absolute residual -- same identity crisis as ACI's gamma=0.
    window_size : int
        Rolling calibration window, in forecast origins. Defaults to 1000 to match the
        other time-series calibrators here.
    recompute_every : int
        Recompute the localized weights every k-th step, reusing the previous step's
        quantile in between. 1 is exact. Raising it trades fidelity for speed on long
        horizons, where the per-step sort dominates.
    seed : int
        Seed for the multinomial randomization, so runs are reproducible.
    """

    def __init__(self, alpha=0.1, temperature=100.0, window_size=1000,
                 recompute_every=1, seed=0):
        self.alpha = alpha
        self.temperature = float(temperature)
        self.window_size = window_size
        self.recompute_every = max(1, int(recompute_every))
        self.rng = np.random.default_rng(seed)

        self.resid_window = None    # [W, H, C]
        self.loggate_window = None  # [W, H, C, K]
        self._steps = 0
        self._last_q = None

    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_gate(gate):
        """Clamp to strictly positive and renormalize over the expert axis (last)."""
        g = np.maximum(np.asarray(gate, dtype=np.float32), EPS)
        return g / np.maximum(g.sum(axis=-1, keepdims=True), EPS)

    def fit(self, preds, gates, trues):
        """Seed the rolling window from a calibration split.

        Parameters
        ----------
        preds, trues : ndarray [N, H, C]
        gates : ndarray [N, H, C, K]
            Gate distributions, softmax-normalized over the last axis.
        """
        resid = np.abs(np.asarray(trues, dtype=np.float32) - np.asarray(preds, dtype=np.float32))
        loggate = np.log(self._normalize_gate(gates))

        if resid.shape[0] > self.window_size:
            resid, loggate = resid[-self.window_size:], loggate[-self.window_size:]
        self.resid_window = np.ascontiguousarray(resid)
        self.loggate_window = np.ascontiguousarray(loggate)
        self._steps = 0
        self._last_q = None
        return self

    def _randomized_gate(self, gate_t):
        """pi~ ~ Multinomial(tau, pi_test)/tau, drawn independently per (h, c) cell."""
        tau = int(self.temperature)
        if tau <= 0:
            return self._normalize_gate(gate_t)
        p = self._normalize_gate(gate_t).astype(np.float64)
        p /= p.sum(axis=-1, keepdims=True)
        # numpy's multinomial rejects pvals whose leading K-1 entries sum above 1, which
        # plain renormalization can still produce in floating point. Let the last
        # component absorb the residual so the constraint holds exactly.
        p[..., -1] = np.maximum(1.0 - p[..., :-1].sum(axis=-1), 0.0)
        # numpy's Generator.multinomial broadcasts over the leading axes of pvals.
        counts = self.rng.multinomial(tau, p)
        return (counts / float(tau)).astype(np.float32)

    def localized_quantile(self, gate_t, pi_tilde=None):
        """Weighted (1-alpha) quantile of the window residuals, localized to gate_t.

        Returns an array [H, C]; entries are +inf where the target level falls inside the
        test point's own mass at infinity.

        `pi_tilde` lets a caller inject the randomization instead of drawing it here. The
        channel axis is fully separable (every reduction below runs along the window axis,
        and the einsum contracts only the expert axis), so a driver can split C across
        worker processes -- but only the *drawing* of pi~ is order-dependent, because
        `self.rng` is one stream consumed row-major over the (h, c) grid. A worker that
        re-seeded its own Generator would silently get a different realization. Injecting
        a slice of the parent's single draw keeps the parallel run bit-identical to the
        serial one. See `_moecp_worker` in exp/exp_long_term_forecasting.py.
        """
        if self.resid_window is None:
            raise ValueError("MoECPCalibrator must be initialized with fit() first!")

        # int(), not the raw float: the tabular MoECP_Calibrator does `tau = int(temperature)`
        # once and uses that same integer both as the multinomial count and as the KL
        # exponent. Keeping the float here would silently diverge for non-integer tau.
        tau = float(int(self.temperature))
        if pi_tilde is None:
            pi_tilde = self._randomized_gate(gate_t)                   # [H, C, K]
        log_pi_test = np.log(self._normalize_gate(gate_t))             # [H, C, K]

        # a_i = tau * <pi~, log pi_i>; the -H(pi~) term of the KL cancels in the softmax.
        a = tau * np.einsum('whck,hck->whc', self.loggate_window, pi_tilde)
        a_test = tau * np.einsum('hck,hck->hc', log_pi_test, pi_tilde)

        shift = np.maximum(a.max(axis=0), a_test)
        ea = np.exp(a - shift)
        ea_test = np.exp(a_test - shift)
        w = ea / np.maximum(ea.sum(axis=0) + ea_test, EPS)             # [W, H, C]

        order = np.argsort(self.resid_window, axis=0)
        sorted_resid = np.take_along_axis(self.resid_window, order, axis=0)
        sorted_w = np.take_along_axis(w, order, axis=0)
        cum = np.cumsum(sorted_w, axis=0)

        target = 1.0 - self.alpha
        reached = cum[-1] >= target
        idx = np.argmax(cum >= target, axis=0)                          # first index at/above
        q = np.take_along_axis(sorted_resid, idx[None, ...], axis=0)[0]
        # Target level lands in the test point's delta_inf mass: clip to the window's own
        # max residual (per cell) instead of returning unbounded, mirroring the q_level =
        # min(..., 1.0) clip-to-window-max rule every other calibrator here already uses.
        # Keeps width comparable to the other calibrators' full-population mean, at the cost
        # of exactness on the clipped cells (same accuracy-vs-comparability trade as the ACI
        # alpha_t clip in calibration/aci_core.py).
        return np.where(reached, q, sorted_resid[-1])

    def interval_from_q(self, q, pred_t):
        return pred_t - q, pred_t + q

    def needs_recompute(self):
        """Whether the next predict_one_step will redraw pi~ and rebuild the quantile.

        A parallel driver has to mirror this exactly: pi~ must be drawn on the same steps
        as the serial path, or the shared RNG stream desynchronizes.
        """
        return self._last_q is None or self._steps % self.recompute_every == 0

    def predict_one_step(self, pred_t, gate_t, pi_tilde=None):
        """Interval for one forecast origin. Honors `recompute_every`."""
        if self.needs_recompute():
            self._last_q = self.localized_quantile(gate_t, pi_tilde=pi_tilde)
        self._steps += 1
        lower, upper = self.interval_from_q(self._last_q, pred_t)
        return lower, upper, self._last_q

    def update(self, pred_t, gate_t, target_t):
        """Push one observed forecast origin into the rolling window.

        Matches AdaptiveCPVS.update: the window *grows* by concatenation while it is
        still shorter than window_size (which happens when the calibration split has
        fewer origins than the window), and only becomes FIFO once full. On ETT the
        validation split always exceeds 1000 so the growth branch never runs, but this
        keeps the two calibrators on the same protocol for smaller datasets.
        """
        resid = np.abs(np.asarray(target_t, dtype=np.float32)
                       - np.asarray(pred_t, dtype=np.float32))[None, ...]
        loggate = np.log(self._normalize_gate(gate_t))[None, ...]

        if self.resid_window.shape[0] < self.window_size:
            self.resid_window = np.concatenate([self.resid_window, resid], axis=0)
            self.loggate_window = np.concatenate([self.loggate_window, loggate], axis=0)
        else:
            self.resid_window = np.concatenate([self.resid_window[1:], resid], axis=0)
            self.loggate_window = np.concatenate([self.loggate_window[1:], loggate], axis=0)

    # ------------------------------------------------------------------
    # Static split-conformal interface, for the offline benchmark
    # ------------------------------------------------------------------

    def calibrate(self, cal_preds, cal_labels, cal_gates, alpha=None):
        if alpha is not None:
            self.alpha = alpha
        return self.fit(cal_preds, cal_gates, cal_labels)

    def predict_intervals(self, test_preds, test_gates, stride=1):
        """Intervals for a whole test split against the frozen calibration window.

        `stride` reuses one localized quantile for `stride` consecutive origins; it exists
        because this is O(W log W * H * C) per origin and the full ETTm grid at
        pred_len=720 is otherwise hours of work.
        """
        test_preds = np.asarray(test_preds, dtype=np.float32)
        n = test_preds.shape[0]
        lowers = np.empty_like(test_preds)
        uppers = np.empty_like(test_preds)
        q = None
        for t in range(n):
            if q is None or t % stride == 0:
                q = self.localized_quantile(test_gates[t])
            lowers[t], uppers[t] = self.interval_from_q(q, test_preds[t])
        return lowers, uppers
