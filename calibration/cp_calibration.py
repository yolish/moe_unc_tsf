import numpy as np

class StandardCP_MSE:
    """
    Standard Conformal Prediction with Sliding Window for MSE-trained models.
    Assumes constant variance (homoscedasticity) implicitly by using unscaled absolute errors.
    """
    def __init__(self, alpha=0.1, window_size=500):
        self.alpha = alpha
        self.window_size = window_size
        self.scores_window = None

    def fit(self, preds, targets):
        """
        Initialize the calibration window with validation data.
        Scores = |y - y_hat|
        """
        scores = np.abs(targets - preds)
        
        if scores.shape[0] > self.window_size:
            self.scores_window = scores[-self.window_size:]
        else:
            self.scores_window = scores
            
    def predict_one_step(self, pred_t):
        """
        Calculate uncertainty interval for a single time step based on past errors.
        Returns: lower_bound, upper_bound, current_q
        """
        if self.scores_window is None:
            raise ValueError("Calibrator must be initialized with fit() first!")

        n = self.scores_window.shape[0]
        
        # Calculate Quantile (Q)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        
        # Calculate Q on the historical errors window
        current_q = np.quantile(self.scores_window, q_level, axis=0, interpolation='higher')
        
        # For Standard CP, the interval width is exactly the Quantile of the errors
        interval_width = current_q 
        
        lower_bound = pred_t - interval_width
        upper_bound = pred_t + interval_width
        
        return lower_bound, upper_bound, current_q

    def update(self, pred_t, target_t):
        """
        Update the sliding window with the new observed error.
        """
        new_score = np.abs(target_t - pred_t)
        
        if new_score.ndim == self.scores_window.ndim - 1:
            new_score = np.expand_dims(new_score, axis=0)
        
        # Sliding Window Logic: Remove oldest, add newest
        if self.scores_window.shape[0] < self.window_size:
            self.scores_window = np.concatenate([self.scores_window, new_score], axis=0)
        else:
            self.scores_window = np.concatenate([self.scores_window[1:], new_score], axis=0)