import numpy as np
import torch
from torch.utils.data import Dataset

class SyntheticDatasetA(Dataset):
    def __init__(self, num_samples=5000, split='train'):
        np.random.seed(42 if split == 'train' else 43)
        self.x = np.random.uniform(-1, 1, num_samples).astype(np.float32)
        self.y = np.zeros_like(self.x)
        
        # Regime 1: x < 0 -> נמשוך את הפונקציה למטה
        mask1 = self.x < 0
        self.y[mask1] = 3 * self.x[mask1] - 2.0 + np.random.normal(0, 0.5, np.sum(mask1)) 
        
        # Regime 2: x >= 0 -> נרים את הפונקציה למעלה
        mask2 = self.x >= 0
        self.y[mask2] = -3 * self.x[mask2] + 2.0 + np.random.normal(0, 0.5, np.sum(mask2))
        
        self.x = torch.from_numpy(self.x).unsqueeze(1)
        self.y = torch.from_numpy(self.y).unsqueeze(1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class SyntheticDatasetB(Dataset):
    def __init__(self, num_samples=10000, split='train'):
        np.random.seed(42 if split == 'train' else 43)
        self.x = np.random.uniform(-1, 1, num_samples).astype(np.float32)
        self.y = np.zeros_like(self.x)
        
        # Regime 1: [-1, -1/3) -> נשאיר את הסינוס
        mask1 = (self.x >= -1) & (self.x < -1/3)
        self.y[mask1] = np.sin(6 * np.pi * self.x[mask1]) + np.random.normal(0, 0.2, np.sum(mask1))
        
        # Regime 2: [-1/3, 1/3) -> נהפוך לקו אופקי גבוה מאוד!
        mask2 = (self.x >= -1/3) & (self.x < 1/3)
        self.y[mask2] = 4.0 + np.random.normal(0, 0.1, np.sum(mask2))
        
        # Regime 3: [1/3, 1] -> נהפוך לקו יורד תלול ונמוך
        mask3 = (self.x >= 1/3) & (self.x <= 1)
        self.y[mask3] = -4 * self.x[mask3] + np.random.normal(0, 0.2, np.sum(mask3))
        
        self.x = torch.from_numpy(self.x).unsqueeze(1)
        self.y = torch.from_numpy(self.y).unsqueeze(1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]