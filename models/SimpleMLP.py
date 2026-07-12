import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.prob_expert = configs.prob_expert
        self.use_quantile_loss = getattr(configs, 'use_quantile_loss', False)
        self.hidden_dim = configs.d_model
        dropout_rate = configs.dropout if hasattr(configs, 'dropout') else 0.1

        # Three hidden layers with ReLU + Dropout(0.1), per the expert architecture in Appendix A.4.1
        self.mlp = nn.Sequential(
            nn.Linear(configs.enc_in, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        if self.use_quantile_loss:
            # 2 outputs = (q_low=0.05, q_high=0.95), trained via pinball loss (see
            # Exp_Tabular_Regression._calculate_loss). Mutually exclusive with prob_expert.
            self.quantile_head = nn.Linear(self.hidden_dim, 2)
        else:
            self.mean_head = nn.Linear(self.hidden_dim, 1)
            if self.prob_expert:
                self.var_head = nn.Linear(self.hidden_dim, 1)

    def forward(self, x):
        h = self.mlp(x)
        if self.use_quantile_loss:
            return self.quantile_head(h)
        mean = self.mean_head(h)
        if self.prob_expert:
            # Softplus limits the variance to be positive
            var = F.softplus(self.var_head(h), threshold=20) + 1e-6
            return mean, var
        return mean