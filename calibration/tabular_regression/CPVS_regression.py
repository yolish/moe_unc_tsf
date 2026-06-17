import torch
import numpy as np

class CP_VS_Static_Calibrator:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.q_val = None

    def fit(self, cal_preds, cal_trues, cal_stds):
        residuals = torch.abs(cal_trues.cpu() - cal_preds.cpu()).squeeze()
        
        normalized_residuals = residuals / (cal_stds.cpu() + 1e-8)
        
        self.q_val = np.quantile(normalized_residuals.numpy(), 1.0 - self.alpha)

    def predict(self, test_preds, test_stds):
        test_preds = test_preds.cpu().squeeze()
        test_stds = test_stds.cpu().squeeze()
        
        margin = self.q_val * test_stds
        intervals = torch.stack([test_preds - margin, test_preds + margin], dim=-1)
        
        return intervals