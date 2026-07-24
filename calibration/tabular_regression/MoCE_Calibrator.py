import torch


class MoCE_Calibrator:
    """
    Mixture of Conformal Experts (MoCE): localized conformal calibration for Soft-MoE regression.

    Unlike MoECP_Calibrator (which conformalizes the *aggregated* prediction's residual, weighted
    by whole-gate-vector KL similarity), MoCE conformalizes each expert's *own* residual
    S_i^(k) = |y_i - mu_k(x_i)| separately, localized by that expert's own (smoothed) gating value
    across the calibration set. Each expert therefore gets its own weighted quantile Q_k, and the
    final threshold is a soft, routing-weighted mixture of all K experts' quantiles -- hedging
    against any single expert's local calibration being unreliable (e.g. in OOD/high-entropy
    regions) rather than relying purely on gate-similarity reweighting.
    """

    def __init__(self, alpha=0.1, epsilon=1e-3):
        self.alpha = alpha
        self.epsilon = float(epsilon)
        self.sorted_residuals = None   # [n,K], per-expert ascending residuals
        self.cumsum_pi_tilde = None    # [n,K], per-expert cumulative smoothed-gate mass
        self.S = None                  # [K], total calibration pi_tilde mass per expert

    def fit(self, cal_expert_preds, cal_trues, cal_gating):
        cal_expert_preds = cal_expert_preds.cpu()          # [n,K]
        cal_trues = cal_trues.cpu().reshape(-1, 1)          # [n,1]
        cal_gating = cal_gating.cpu()                       # [n,K], raw pi_k(x_i)

        pi_tilde = (cal_gating + self.epsilon) / (cal_gating + self.epsilon).sum(dim=1, keepdim=True)  # Eq. 2
        residuals = torch.abs(cal_trues - cal_expert_preds)  # Eq. 1, [n,K]

        sorted_residuals, sort_idx = torch.sort(residuals, dim=0)
        sorted_pi_tilde = torch.gather(pi_tilde, 0, sort_idx)
        cumsum_pi_tilde = torch.cumsum(sorted_pi_tilde, dim=0)

        self.sorted_residuals = sorted_residuals
        self.cumsum_pi_tilde = cumsum_pi_tilde
        self.S = cumsum_pi_tilde[-1]

    def predict(self, test_expert_preds, test_gating):
        if self.sorted_residuals is None:
            raise ValueError("Calibrator must be fitted before calling predict.")

        test_expert_preds = test_expert_preds.cpu()  # [m,K]
        test_gating = test_gating.cpu()               # [m,K], raw pi_k(x_test)

        n, K = self.sorted_residuals.shape
        m = test_gating.shape[0]

        pi_tilde_test = (test_gating + self.epsilon) / (test_gating + self.epsilon).sum(dim=1, keepdim=True)  # Eq. 2

        Q = torch.empty(m, K)
        for k in range(K):
            denom_k = self.S[k] + pi_tilde_test[:, k]           # Eq. 3 denominator
            target_k = (1.0 - self.alpha) * denom_k
            idx_k = torch.searchsorted(self.cumsum_pi_tilde[:, k].contiguous(), target_k)

            inf_mask = idx_k >= n
            idx_k_clamped = torch.clamp(idx_k, max=n - 1)
            q_k = self.sorted_residuals[idx_k_clamped, k]
            # Target quantile falls on the calibration weights' point mass at delta_{+inf}: the
            # missing probability mass belongs to the reserved test-point weight, so the interval
            # must be unbounded here rather than clipped to the max observed residual (same
            # weighted-conformal argument as MoECP_Calibrator).
            Q[:, k] = torch.where(inf_mask, torch.full_like(q_k, float('inf')), q_k)

        mu_hat = torch.sum(test_gating * test_expert_preds, dim=1)  # Eq. 4 (raw gate)
        # 0 * inf = nan in IEEE arithmetic: a zero-weight expert must contribute nothing to the
        # threshold regardless of whether its own (possibly infinite) local quantile is defined.
        contrib = torch.where(test_gating > 0, test_gating * Q, torch.zeros_like(Q))
        threshold = torch.sum(contrib, dim=1)                         # Eq. 5 (raw gate)

        return torch.stack([mu_hat - threshold, mu_hat + threshold], dim=1)
