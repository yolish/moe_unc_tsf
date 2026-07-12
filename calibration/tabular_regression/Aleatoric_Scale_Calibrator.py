import numpy as np
import torch

class AleatoricScaleCalibrator:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.q_sq = None

    def fit(self, cal_preds, cal_trues, cal_stds_aleat, cal_stds_epist):
        """
        Learn the scaling factor q^2 from the calibration set.
        """
        residuals_sq = (cal_trues.squeeze() - cal_preds.squeeze()) ** 2
        epist_var = cal_stds_epist.squeeze() ** 2
        aleat_var = cal_stds_aleat.squeeze() ** 2 + 1e-8 # Prevent division by zero
        
        # Calculate conformity scores
        scores = (residuals_sq - epist_var) / aleat_var
        scores = torch.clamp(scores, min=0.0).numpy()
        
        # Compute the empirical conformal quantile
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(max(q_level, 0.0), 1.0)
        
        self.q_sq = np.quantile(scores, q_level, method='higher')

    def predict(self, test_preds, test_stds_aleat, test_stds_epist):
        """
        Construct prediction intervals using the learned scaling factor.
        """
        if self.q_sq is None:
            raise ValueError("Calibrator must be fitted before calling predict.")
            
        test_aleat_var = test_stds_aleat.squeeze() ** 2
        test_epist_var = test_stds_epist.squeeze() ** 2
        
        # Adjust only the aleatoric component using q_sq
        calibrated_std = torch.sqrt(self.q_sq * test_aleat_var + test_epist_var)
        
        lower_bound = test_preds.squeeze() - calibrated_std
        upper_bound = test_preds.squeeze() + calibrated_std
        
        intervals = torch.stack([lower_bound, upper_bound], dim=1)
        return intervals