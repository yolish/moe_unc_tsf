import torch
import numpy as np
import torch.nn.functional as F

class MoECP_Calibrator:
    def __init__(self, alpha=0.1, temperature=1.0):
        self.alpha = alpha
        self.temperature = temperature
        self.cal_residuals = None
        self.cal_gating_weights = None

    def fit(self, cal_preds, cal_trues, cal_gating_weights):
        self.cal_residuals = torch.abs(cal_trues.cpu() - cal_preds.cpu()).squeeze()
        self.cal_gating_weights = cal_gating_weights.cpu()

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
            kl_div = F.kl_div(
                torch.log(self.cal_gating_weights + 1e-8), 
                pi_tilde.unsqueeze(0).expand_as(self.cal_gating_weights), 
                reduction='none'
            ).sum(dim=1)
            
            weights = torch.exp(-self.temperature * kl_div)
            # נרמול: הפלוס 1.0 מייצג את משקל נקודת הטסט עצמה (משוואה 5)
            weights = weights / (weights.sum() + 1.0) 
            
            sorted_residuals, indices = torch.sort(self.cal_residuals)
            sorted_weights = weights[indices]
            
            cumulative_weights = torch.cumsum(sorted_weights, dim=0)
            
            target_prob = 1.0 - self.alpha
            quantile_idx = torch.searchsorted(cumulative_weights, target_prob).item()
            
            # מניעת חריגה ממערך השאריות אם ההסתברות נופלת בדיוק על נקודת הטסט
            if quantile_idx >= len(sorted_residuals):
                q_val = sorted_residuals[-1]
            else:
                q_val = sorted_residuals[quantile_idx]
            
            intervals.append([test_preds[i] - q_val, test_preds[i] + q_val])
            
        return torch.tensor(intervals)