import numpy as np

class AdaptiveCPVS:
    def __init__(self, alpha=0.1, window_size=500):
        self.alpha = alpha
        self.window_size = window_size
        self.scores_window = None  # Buffer to hold recent non-conformity scores

    def fit(self, preds, sigma, targets):
        scores = np.abs(targets - preds) / (sigma + 1e-8)
        
        if scores.shape[0] > self.window_size:
            self.scores_window = scores[-self.window_size:]
        else:
            self.scores_window = scores
            
    def predict_one_step(self, pred_t, sigma_t):

        if self.scores_window is None:
            raise ValueError("Calibrator must be initialized with fit() first!")

        n = self.scores_window.size 
        
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        
        current_q = np.quantile(self.scores_window, q_level, interpolation='linear')
        
        interval_width = current_q * sigma_t
        
        lower_bound = pred_t - interval_width
        upper_bound = pred_t + interval_width
        
        return lower_bound, upper_bound, current_q

    def update(self, pred_t, sigma_t, target_t):

        new_score = np.abs(target_t - pred_t) / (sigma_t + 1e-8)
        
        new_score = np.expand_dims(new_score, axis=0)
        
        if self.scores_window.shape[0] < self.window_size:
            self.scores_window = np.concatenate([self.scores_window, new_score], axis=0)
            
        else:
            self.scores_window = np.concatenate([self.scores_window[1:], new_score], axis=0)