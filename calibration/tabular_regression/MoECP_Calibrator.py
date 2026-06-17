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
            
            kl_div = F.kl_div(
                torch.log(self.cal_gating_weights + 1e-8), 
                pi_test.unsqueeze(0).expand_as(self.cal_gating_weights), 
                reduction='none'
            ).sum(dim=1)
            
            weights = torch.exp(-self.temperature * kl_div)
            weights = weights / (weights.sum() + 1.0)
            
            sorted_residuals, indices = torch.sort(self.cal_residuals)
            sorted_weights = weights[indices]
            
            cumulative_weights = torch.cumsum(sorted_weights, dim=0)
            
            target_prob = 1.0 - self.alpha
            quantile_idx = torch.searchsorted(cumulative_weights, target_prob)
            
            if quantile_idx >= len(sorted_residuals):
                quantile_idx = len(sorted_residuals) - 1
                
            q_val = sorted_residuals[quantile_idx]
            
            intervals.append([test_preds[i] - q_val, test_preds[i] + q_val])
            
        return torch.tensor(intervals)