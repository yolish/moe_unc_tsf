import torch


class MoECP_Calibrator:
    def __init__(self, alpha=0.1, temperature=100.0):
        self.alpha = alpha
        self.temperature = float(temperature)
        self.cal_residuals = None
        self.cal_gating_weights = None

    def fit(self, cal_preds, cal_trues, cal_gating_weights):
        cal_preds = cal_preds.cpu().squeeze()
        cal_trues = cal_trues.cpu().squeeze()
        self.cal_gating_weights = cal_gating_weights.cpu()
        
        self.cal_residuals = torch.abs(cal_trues - cal_preds)

    def predict(self, test_preds, test_gating_weights):
        test_preds = test_preds.cpu().squeeze()
        test_gating_weights = test_gating_weights.cpu()
        intervals = []

        for i in range(len(test_preds)):
            pi_test = test_gating_weights[i]
            
            # --- שלב הרנדומיזציה (לפי משוואה 3 במאמר) ---
            tau = int(self.temperature)
            if tau > 0:
                # הבטחה שההסתברויות חיוביות לחלוטין ומסתכמות ל-1 עבור דגימה מולטינומית
                pi_test_probs = torch.clamp(pi_test, min=1e-8)
                pi_test_probs = pi_test_probs / pi_test_probs.sum()
                
                # דגימה מתוך התפלגות מולטינומית
                L = torch.distributions.Multinomial(total_count=tau, probs=pi_test_probs).sample()
                pi_tilde = L / tau
            else:
                pi_tilde = pi_test
            # ---------------------------------------------
            
            # חישוב KL Divergence מול נקודת הטסט המורעשת (pi_tilde)
            kl_div = torch.nn.functional.kl_div(
                torch.log(self.cal_gating_weights + 1e-8), 
                pi_tilde.unsqueeze(0).expand_as(self.cal_gating_weights), 
                reduction='none'
            ).sum(dim=1)
            
            weights = torch.exp(-kl_div * tau)

            # Test point's own weight w_{n+1} = exp(-tau * D(pi_tilde, pi_test)):
            test_kl_div = torch.nn.functional.kl_div(
                torch.log(pi_test + 1e-8),
                pi_tilde,
                reduction='sum'
            )
            test_weight = torch.exp(-test_kl_div * tau)

            # π(X_{n+1}) is the *original* (non-randomized) test gate here, distinct
            # from π̃ above - Algorithm 1 step 4 defines w_{n+1}=exp(-τD(π̃,π(X_{n+1})))
            # for i=n+1 the same way as for calibration points, so this is generally
            # != 1 (see Theorem 1's proof, not a special-cased constant).
            weights = weights / (weights.sum() + test_weight)

            sorted_residuals, indices = torch.sort(self.cal_residuals)
            sorted_weights = weights[indices]
            
            cumulative_weights = torch.cumsum(sorted_weights, dim=0)
            
            target_prob = 1.0 - self.alpha
            quantile_idx = torch.searchsorted(cumulative_weights, target_prob).item()
            
            # Target quantile falls on the calibration weights' point mass at delta_{+inf}
            # (Algorithm 1 / Theorem 1): the interval must be unbounded here, not clipped to
            # the max observed residual, otherwise the finite-sample coverage guarantee can fail.
            if quantile_idx >= len(sorted_residuals):
                q_val = torch.tensor(float('inf'))
            else:
                q_val = sorted_residuals[quantile_idx]
            
            intervals.append([test_preds[i] - q_val, test_preds[i] + q_val])

        return torch.tensor(intervals)