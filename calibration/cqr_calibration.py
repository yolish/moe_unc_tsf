import numpy as np

class OnlineCQRQuantile:
    def __init__(self, alpha=0.1, window_size=500):
        self.alpha = alpha
        self.window_size = window_size
        self.scores_window = None

    def fit(self, lower_preds, upper_preds, targets):
        scores = np.maximum(lower_preds - targets, targets - upper_preds)
        
        if scores.shape[0] > self.window_size:
            self.scores_window = scores[-self.window_size:]
        else:
            self.scores_window = scores
            
    def predict_one_step(self, lower_t, upper_t):
        if self.scores_window is None:
            raise ValueError("Calibrator must be initialized with fit() first!")

        n = self.scores_window.shape[0]
        
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        
        q_hat = np.quantile(self.scores_window, q_level, axis=0, interpolation='higher')
        
        calibrated_lower = lower_t - q_hat
        calibrated_upper = upper_t + q_hat
        
        return calibrated_lower, calibrated_upper, q_hat

    def update(self, lower_t, upper_t, target_t):
        new_score = np.maximum(lower_t - target_t, target_t - upper_t)
        
        if self.scores_window is not None:
            if new_score.ndim == self.scores_window.ndim - 1:
                new_score = np.expand_dims(new_score, axis=0)
            elif new_score.ndim == 0 and self.scores_window.ndim == 1:
                 new_score = np.expand_dims(new_score, axis=0)
        
        if self.scores_window.shape[0] < self.window_size:
            self.scores_window = np.concatenate([self.scores_window, new_score], axis=0)
        else:
            self.scores_window = np.concatenate([self.scores_window[1:], new_score], axis=0)