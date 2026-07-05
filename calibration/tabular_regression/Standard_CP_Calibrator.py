import numpy as np
import torch

class Standard_CP_Calibrator:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.q = None

    def fit(self, cal_preds, cal_trues):
        # וידוא שאנחנו עובדים עם Numpy Arrays
        if torch.is_tensor(cal_preds): cal_preds = cal_preds.cpu().numpy()
        if torch.is_tensor(cal_trues): cal_trues = cal_trues.cpu().numpy()
        
        cal_preds = cal_preds.squeeze()
        cal_trues = cal_trues.squeeze()
        
        # 1. חישוב השאריות (Absolute Residuals) - ההבדל בין התחזית לאמת
        residuals = np.abs(cal_trues - cal_preds)
        
        # 2. מציאת האחוזון (Quantile) המתאים
        n = len(residuals)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        
        # 3. שמירת האחוזון (הוא יהיה הרוחב הקבוע לכל התחזיות)
        self.q = np.quantile(residuals, q_level, method='higher')

    def predict(self, test_preds):
        if torch.is_tensor(test_preds): test_preds = test_preds.cpu().numpy()
        test_preds = test_preds.squeeze()
        
        # יצירת המרווחים: התחזית ± האחוזון שחושב
        lower_bound = test_preds - self.q
        upper_bound = test_preds + self.q
        
        return np.stack([lower_bound, upper_bound], axis=1)