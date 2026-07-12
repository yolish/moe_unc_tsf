import torch
import numpy as np

class CP_VS_Static_Calibrator:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.q_val = None

    def fit(self, cal_preds, cal_trues, cal_stds):
        residuals = torch.abs(cal_trues.cpu().squeeze() - cal_preds.cpu().squeeze())
        normalized_residuals = residuals / (cal_stds.cpu() + 1e-8)
        
        scores = normalized_residuals.numpy()
        n = len(scores)
        
        # Finite Sample Correction (כדי להבטיח כיסוי מתמטי מדויק)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(max(q_level, 0.0), 1.0)
        
        # מתודת 'higher' מבטיחה שאנחנו לוקחים את הערך השמרני כדי להבטיח את הכיסוי
        self.q_val = np.quantile(scores, q_level, method='higher')

    def predict(self, test_preds, test_stds):
        test_preds = test_preds.cpu().squeeze()
        test_stds = test_stds.cpu().squeeze()
        
        margin = self.q_val * test_stds
        intervals = torch.stack([test_preds - margin, test_preds + margin], dim=-1)
        
        return intervals