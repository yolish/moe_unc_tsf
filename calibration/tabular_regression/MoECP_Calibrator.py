import torch
from torch.distributions import Multinomial
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
        
        # --- תחילת קוד הלוגים: משתני מעקב ---
        import numpy as np
        log_effective_samples = []
        log_min_kl = []
        log_routing_confidence = []
        # --- סוף קוד הלוגים ---
        
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
            
            # --- תחילת התיקון ---
            # חישוב ה-KL Divergence בין ההסתברות המורעשת (pi_tilde) להסתברות המקורית של הטסט (pi_test)
            test_kl_div = torch.nn.functional.kl_div(
                torch.log(pi_test + 1e-8), 
                pi_tilde, 
                reduction='sum'
            )
            # חישוב המשקל האמיתי של נקודת הטסט
            test_weight = torch.exp(-test_kl_div * tau)

            
            # נרמול עם המשקל האמיתי במקום 1.0 קשיח
            weights = weights / (weights.sum() + test_weight) 
            # --- סוף התיקון ---
            
            # --- תחילת קוד הלוגים: איסוף נתונים לכל דגימה ---
            effective_points = torch.sum(weights > 0.01).item()
            log_effective_samples.append(effective_points)
            log_min_kl.append(torch.min(kl_div).item())
            log_routing_confidence.append(torch.max(pi_test).item())
            # --- סוף קוד הלוגים ---
            
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
            
        # --- תחילת קוד הלוגים: הדפסת דוח ניתוח ---
        avg_effective = np.mean(log_effective_samples)
        median_effective = np.median(log_effective_samples)
        avg_confidence = np.mean(log_routing_confidence)
        
        print("\n" + "="*40)
        print("🔍 MoECP Inner Mechanics Analysis 🔍")
        print(f"Routing Confidence (Max Weight Avg): {avg_confidence:.4f}")
        print(f"Avg Effective Calib Points Used (weight > 1%): {avg_effective:.1f} out of {len(self.cal_residuals)}")
        print(f"Median Effective Calib Points: {median_effective:.1f}")
        print(f"Avg Minimum KL Divergence: {np.mean(log_min_kl):.4f}")
        print("="*40 + "\n")
        # --- סוף קוד הלוגים ---
            
        return torch.tensor(intervals)