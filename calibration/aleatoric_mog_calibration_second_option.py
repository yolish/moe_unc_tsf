import numpy as np
import numpy as np

class AleatoricMOGCalibratorSecondOption:
    def __init__(self, alpha=0.1, window_size=1000):
        self.alpha = alpha
        self.window_size = window_size
        self.scores_window = None
        
    def _calc_heuristic(self, ale_var, epi_var):
        std_ale = np.sqrt(np.maximum(0, ale_var))
        std_epi = np.sqrt(np.maximum(0, epi_var))
        var_ale = np.maximum(0, ale_var)
        var_epi = np.maximum(0, epi_var)
        
        # חישוב היחס: אפיסטמי חלקי אליאטורי
        # מוסיפים 1e-8 למניעת חלוקה באפס
        ratio = var_epi / (var_ale + 1e-8)
        
        # חותכים את היחס המקסימלי ל-5 כדי למנוע התפוצצות נומרית של ה-exp
        ratio_clipped = np.clip(ratio, a_min=0, a_max=5.0)
        
        # חישוב ההיוריסטיקה: סטיית תקן אליאטורית כפול האקספוננט של היחס
        H_x = (std_epi + std_ale) * np.exp(ratio_clipped)
        return H_x

    def fit(self, val_preds, val_ale, val_epi, val_trues):
        H_x = self._calc_heuristic(val_ale, val_epi)
        
        # חישוב הציונים
        scores = np.abs(val_trues - val_preds) / (H_x + 1e-8)
        
        if len(scores) > self.window_size:
            self.scores_window = scores[-self.window_size:].copy()
        else:
            self.scores_window = scores.copy()

    def predict_one_step(self, test_pred, test_ale, test_epi):
        n = self.scores_window.shape[0]
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        
        # מציאת ה-q מתוך חלון ה-Score
        q = np.quantile(self.scores_window, q_level, axis=0, interpolation='higher')
        
        # חישוב ההיוריסטיקה לדוגמה החדשה
        H_x_test = self._calc_heuristic(test_ale, test_epi)
        
        # חישוב רוחב רווח הסמך
        width = q * H_x_test
        
        lower = test_pred - width
        upper = test_pred + width
        
        return lower, upper, q

    def update(self, test_pred, test_ale, test_epi, test_true):
        H_x = self._calc_heuristic(test_ale, test_epi)
        new_score = np.abs(test_true - test_pred) / (H_x + 1e-8)
        
        self.scores_window = np.roll(self.scores_window, -1, axis=0)
        self.scores_window[-1] = new_score