import numpy as np

class ConformalCalibrator:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.q = None 

    def fit(self, preds, sigma, targets):
        scores = np.abs(targets - preds) / (sigma + 1e-8)

        scores = scores.flatten()
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        
        self.q = np.quantile(scores, q_level, interpolation='higher')
        
        return self.q

    def predict(self, preds, sigma):
        if self.q is None:
            raise ValueError("Calibrator must be fitted on validation data first!")
        interval_width = self.q * sigma 
        lower_bound = preds - interval_width
        upper_bound = preds + interval_width
        
        return lower_bound, upper_bound