import numpy as np
import torch

class AleatoricCPVSCalibrator:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.q_hat = None

    def fit(self, cal_preds, cal_trues, cal_stds_aleat):
        """
        Learn the conformity score threshold (q_hat) based solely on aleatoric uncertainty.
        """
        residuals = torch.abs(cal_trues.squeeze() - cal_preds.squeeze())
        stds = cal_stds_aleat.squeeze() + 1e-8 # Prevent division by zero
        
        # Calculate CP-VS conformity scores: |y - y_hat| / std_aleatoric
        scores = (residuals / stds).numpy()
        
        # Compute the empirical conformal quantile
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(max(q_level, 0.0), 1.0)
        
        self.q_hat = np.quantile(scores, q_level)

    def predict(self, test_preds, test_stds_aleat):
        """
        Construct prediction intervals using the learned q_hat and aleatoric uncertainty.
        """
        if self.q_hat is None:
            raise ValueError("Calibrator must be fitted before calling predict.")
            
        margin = self.q_hat * test_stds_aleat.squeeze()
        
        lower_bound = test_preds.squeeze() - margin
        upper_bound = test_preds.squeeze() + margin
        
        intervals = torch.stack([lower_bound, upper_bound], dim=1)
        return intervals