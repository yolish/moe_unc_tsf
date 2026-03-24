import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.prob_expert = configs.prob_expert
        self.hidden_dim = configs.d_model
        
        self.mlp = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )
        self.mean_head = nn.Linear(self.hidden_dim, 1)
        if self.prob_expert:
            self.var_head = nn.Linear(self.hidden_dim, 1)

    def forward(self, x):
        h = self.mlp(x)
        mean = self.mean_head(h)
        if self.prob_expert:
            # Softplus limits the variance to be positive
            var = F.softplus(self.var_head(h), threshold=20) + 1e-6
            return mean, var
        return mean