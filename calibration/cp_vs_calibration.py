import numpy as np

class CPVSHorizonCalibration:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.q = None 

    def fit(self, preds, sigma, targets):
        scores = np.abs(targets - preds) / (sigma + 1e-8)

        # Change 1: Remove the flattening line to preserve time dimension (Horizon)
        # scores = scores.flatten()  <-- Delete or comment out this line
        
        # Change 2: n is now the number of samples (Batch Size), not total elements
        n = scores.shape[0]  
        
        # q_level calculation remains the same, but the value changes due to new n
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        
        # Change 3: Compute Quantile for each column (time step) separately
        self.q = np.quantile(scores, q_level, axis=0, interpolation='higher')
        
        return self.q

    def predict(self, preds, sigma):
        if self.q is None:
            raise ValueError("Calibrator must be fitted on validation data first!")
        interval_width = self.q * sigma 
        lower_bound = preds - interval_width
        upper_bound = preds + interval_width
        
        return lower_bound, upper_bound