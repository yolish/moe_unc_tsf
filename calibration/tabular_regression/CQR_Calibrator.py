import numpy as np
import torch

class CQR_Calibrator:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.q_hat = None

    def fit(self, cal_lower, cal_upper, cal_trues):
        if torch.is_tensor(cal_lower): cal_lower = cal_lower.cpu().numpy()
        if torch.is_tensor(cal_upper): cal_upper = cal_upper.cpu().numpy()
        if torch.is_tensor(cal_trues): cal_trues = cal_trues.cpu().numpy()
        cal_lower, cal_upper, cal_trues = cal_lower.squeeze(), cal_upper.squeeze(), cal_trues.squeeze()

        # CQR conformity score (same as calibration/cqr_calibration.py::OnlineCQRQuantile)
        scores = np.maximum(cal_lower - cal_trues, cal_trues - cal_upper)
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(max(q_level, 0.0), 1.0)
        self.q_hat = np.quantile(scores, q_level, method='higher')

    def predict(self, test_lower, test_upper):
        if torch.is_tensor(test_lower): test_lower = test_lower.cpu().numpy()
        if torch.is_tensor(test_upper): test_upper = test_upper.cpu().numpy()
        test_lower, test_upper = test_lower.squeeze(), test_upper.squeeze()
        return np.stack([test_lower - self.q_hat, test_upper + self.q_hat], axis=1)
