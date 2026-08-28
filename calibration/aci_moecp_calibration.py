import numpy as np

from calibration.aci_core import ACIState
from calibration.moecp_calibration import EPS, MoECPCalibrator


class ACIMoECPCalibrator(MoECPCalibrator):
    """MoECP whose target level is driven by Adaptive Conformal Inference.

    Score and interval are those of MoECPCalibrator -- the raw absolute residual, with the
    adaptivity carried by a *gate-localized* weighted quantile rather than by rescaling the
    score. What changes is the level that quantile is read off at: the base reads the
    weighted CDF at the fixed 1 - alpha, this reads it at a per-(horizon, channel)
    1 - alpha_t driven by the recursion of Gibbs & Candes (NeurIPS 2021).

    gamma = 0 recovers MoECPCalibrator bit-for-bit.

    Subclass, not a copy. The expensive and delicate parts -- the randomized gate draw with
    its pvals-renormalization workaround, the KL-cancellation einsum/softmax, the RNG
    ownership contract with `_moecp_worker`, `fit`, `update` -- are inherited untouched, so
    they cannot drift from the base. Only `localized_quantile` is overridden, and only to
    make one line per-cell (see `_read_off`).

    The +inf branch, and why it is clipped anyway
    ---------------------------------------------
    Until 2026-08-14 this class returned +inf when the target level fell inside the test
    point's own mass at infinity -- i.e. when the level was too demanding for the localized
    window to express -- and that was deliberate. An unbounded interval cannot be missed, so
    err_t = 0, so alpha_t rises, so the level drops until the CDF criterion is reachable and
    the interval returns bounded, within roughly 1/gamma origins. That is the paper's own
    restoring force arriving through the +inf branch instead of through an unbounded alpha_t,
    and it is what spared this calibrator the integrator windup its siblings suffer (a cell
    whose window cannot cover it parks at alpha_t = 0, keeps missing, and nothing pushes it
    back -- see ACIAleatoricScaleCalibrator's docstring).

    That branch now clips to the window's max residual instead, matching the base
    MoECPCalibrator after its own clip-to-window-max change of 2026-08-13. The reason is
    comparability, not correctness: base and +ACI MoECP rows are reported side by side, and
    a width averaged over finite-only cells cannot be compared against one averaged over
    every cell. The cost is real and is the same trade the alpha_t clip makes -- on cells the
    localized window genuinely cannot cover, the restoring force above is gone, so those
    cells can now wind down to alpha_t = 0 like the siblings. Watch the driver's `Unbounded:`
    line: it counts how often the clip fires (pre-clip it counted true +inf intervals), and a
    large value means a correspondingly large share of cells lost the correction.
    """

    def __init__(self, alpha=0.1, gamma=0.01, temperature=1.0, window_size=1000,
                 recompute_every=1, seed=0):
        super().__init__(alpha=alpha, temperature=temperature, window_size=window_size,
                         recompute_every=recompute_every, seed=seed)
        self.aci = ACIState(alpha=alpha, gamma=gamma)
        self._cdf_cache = None

    def fit(self, preds, gates, trues):
        super().fit(preds, gates, trues)
        self.aci.reset(self.resid_window.shape[1:])
        self._cdf_cache = None
        return self

    # ------------------------------------------------------------------
    # localized_quantile, split so the alpha_t-dependent half can run every origin.
    # Together these reproduce MoECPCalibrator.localized_quantile line for line; the only
    # change is `target`, which is a scalar 1 - alpha in the base (moecp_calibration.py:180)
    # and a per-(H, C) array here. Everything else is verbatim -- diff the two by eye if you
    # touch either.
    # ------------------------------------------------------------------

    def _localized_cdf(self, gate_t, pi_tilde=None):
        """The expensive half: draw pi~, build the localized weights, sort, accumulate.

        Independent of alpha_t, so it may be cached across `recompute_every` origins. This
        is also the half that consumes `self.rng`, so it must be called on exactly the same
        schedule as the base's `localized_quantile` or the RNG stream desynchronizes.
        """
        if self.resid_window is None:
            raise ValueError("MoECPCalibrator must be initialized with fit() first!")

        tau = float(int(self.temperature))
        if pi_tilde is None:
            pi_tilde = self._randomized_gate(gate_t)
        log_pi_test = np.log(self._normalize_gate(gate_t))

        a = tau * np.einsum('whck,hck->whc', self.loggate_window, pi_tilde)
        a_test = tau * np.einsum('hck,hck->hc', log_pi_test, pi_tilde)

        shift = np.maximum(a.max(axis=0), a_test)
        ea = np.exp(a - shift)
        ea_test = np.exp(a_test - shift)
        w = ea / np.maximum(ea.sum(axis=0) + ea_test, EPS)

        order = np.argsort(self.resid_window, axis=0)
        sorted_resid = np.take_along_axis(self.resid_window, order, axis=0)
        sorted_w = np.take_along_axis(w, order, axis=0)
        cum = np.cumsum(sorted_w, axis=0)

        return cum, sorted_resid

    @staticmethod
    def _read_off(cum, sorted_resid, target):
        """The cheap half: read the weighted CDF at `target`.

        `target` is [H, C] here where the base has a scalar, which is the whole ACI change.
        The comparisons broadcast: cum is [W, H, C], so `cum >= target` is [W, H, C] and
        `cum[-1] >= target` is [H, C], exactly as with a scalar.

        The cast is load-bearing. `cum` is float32 and the base compares it against a Python
        float, which numpy value-casts down to float32; an un-cast float64 target array would
        instead promote `cum` to float64 and flip comparisons on cells sitting within one
        float32 ulp of the level -- a silent divergence from the base at gamma = 0.
        """
        target = np.asarray(target, dtype=cum.dtype)

        reached = cum[-1] >= target
        idx = np.argmax(cum >= target, axis=0)
        q = np.take_along_axis(sorted_resid, idx[None, ...], axis=0)[0]
        # Clip to the window's own max residual rather than returning +inf, matching the
        # 2026-08-13 clip-to-window-max rule in MoECPCalibrator.localized_quantile. The two
        # must agree: base and +ACI MoECP rows sit in the same report, and a width averaged
        # over finite-only cells is not comparable to one averaged over every cell. See the
        # class docstring for what this costs the recursion.
        return np.where(reached, q, sorted_resid[-1])

    def localized_quantile(self, gate_t, pi_tilde=None):
        cum, sorted_resid = self._localized_cdf(gate_t, pi_tilde=pi_tilde)
        return self._read_off(cum, sorted_resid, self.aci.level)

    def interval_from_q(self, q, pred_t):
        """Stateful, unlike the base: records the level served for the delayed ACI update."""
        self.aci.record(q)
        return pred_t - q, pred_t + q

    def predict_one_step(self, pred_t, gate_t, pi_tilde=None):
        # Expensive half on the recompute_every schedule (and on the same steps as the base,
        # which is what keeps the RNG stream aligned); cheap half every origin, because
        # alpha_t moved and a cached q would freeze the recursion.
        if self.needs_recompute():
            self._cdf_cache = self._localized_cdf(gate_t, pi_tilde=pi_tilde)
        self._steps += 1

        q = self._read_off(*self._cdf_cache, self.aci.level)
        self._last_q = q
        lower, upper = self.interval_from_q(q, pred_t)
        return lower, upper, q

    def update(self, pred_t, gate_t, target_t):
        resid = np.abs(np.asarray(target_t, dtype=np.float32)
                       - np.asarray(pred_t, dtype=np.float32))

        # Score against the level actually served at this origin *before* the observation
        # joins the window. A +inf q scores as covered by construction -- see the class
        # docstring; that is the restoring force, not a case to mask out.
        self.aci.step(resid, fallback_q=self._last_q)

        super().update(pred_t, gate_t, target_t)
