import numpy as np

class OnlineCQRQuantile:
    def __init__(self, alpha=0.1, window_size=300):
        self.alpha = alpha
        self.window_size = window_size
        self.scores_window = None

    def fit(self, lower_preds, upper_preds, targets):
        # חישוב שגיאת הכיול הראשונית (Non-conformity scores)
        scores = np.maximum(lower_preds - targets, targets - upper_preds)
        
        # שמירה בחלון ההיסטוריה
        if scores.shape[0] > self.window_size:
            self.scores_window = scores[-self.window_size:]
        else:
            self.scores_window = scores
            
    def predict_one_step(self, lower_t, upper_t):
        if self.scores_window is None:
            raise ValueError("Calibrator must be initialized with fit() first!")

        # n = מספר הדוגמאות בחלון ההיסטוריה
        n = self.scores_window.shape[0]
        
        # חישוב רמת ה-Quantile הנדרשת (תיקון CQR)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        
        # חישוב ה-q_hat (תיקון סקלר או מטריצה, תלוי במימדי החלון)
        # שימוש ב-axis=0 מבטיח חישוב לכל נקודת זמן/פיצ'ר בנפרד אם המימדים מאפשרים זאת,
        # או חישוב גלובלי אם רוצים. כאן נשתמש בברירת המחדל לחיזוי חזק.
        # כדי לפשט: אם scores_window הוא [W, 96, 7], התוצאה תהיה סקלר גלובלי שמכסה הכל
        # או שניתן לחשב פר-מימד. בקוד המקורי השתמשנו ב-np.quantile גלובלי או משוער.
        # כדי למנוע קריסות נשתמש בחישוב גלובלי על כל החלון (שמרני ובטוח):
        q_hat = np.quantile(self.scores_window, q_level, interpolation='higher')
        
        calibrated_lower = lower_t - q_hat
        calibrated_upper = upper_t + q_hat
        
        return calibrated_lower, calibrated_upper, q_hat

    def update(self, lower_t, upper_t, target_t):
        new_score = np.maximum(lower_t - target_t, target_t - upper_t)
        
        # --- תיקון מימדים קריטי ---
        if self.scores_window is not None:
            # אם ה-window הוא תלת מימדי (למשל [Window, 96, 7]) והציון החדש הוא דו-מימדי ([96, 7])
            # צריך להוסיף לו מימד בהתחלה -> [1, 96, 7]
            if new_score.ndim == self.scores_window.ndim - 1:
                new_score = np.expand_dims(new_score, axis=0)
            elif new_score.ndim == 0 and self.scores_window.ndim == 1:
                 new_score = np.expand_dims(new_score, axis=0)
        elif np.ndim(new_score) == 0:
             # טיפול במקרה של סקלר בודד באתחול
             new_score = np.expand_dims(new_score, axis=0)
        # --------------------------
        
        if self.scores_window.shape[0] < self.window_size:
            self.scores_window = np.concatenate([self.scores_window, new_score], axis=0)
        else:
            self.scores_window = np.concatenate([self.scores_window[1:], new_score], axis=0)