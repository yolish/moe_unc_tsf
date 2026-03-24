import torch
import torch.nn as nn
import torch.nn.functional as F

class Gating(nn.Module): 
    def __init__(self, configs):
        super(Gating, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.num_experts)
        )

    def forward(self, x):
        logits = self.net(x)
        weights = F.softmax(logits, dim=-1)
        return weights

class Model(nn.Module):
    def __init__(self, configs, expert_model):
        super(Model, self).__init__()
        self.num_experts = configs.num_experts
        self.unc_gating = configs.unc_gating
        self.prob_expert = configs.prob_expert
        
        if self.unc_gating:
            assert self.prob_expert, "MoGU (unc_gating) must be used with probabilistic experts (--prob_expert)!"
            
        self.experts = nn.ModuleList([expert_model(configs) for _ in range(self.num_experts)])
        
        if not self.unc_gating:
            self.gating = Gating(configs)

    def forward(self, x):
        expert_out = []
        expert_unc = []
        
        for expert in self.experts:
            if self.prob_expert:
                mean, var = expert(x)
                expert_unc.append(var)
            else:
                mean = expert(x)
            expert_out.append(mean)
            
        expert_out = torch.stack(expert_out, dim=1)
        
        if self.prob_expert:
            expert_unc = torch.stack(expert_unc, dim=1)
            
        if self.unc_gating:
            # MoGU: Routing weights are inversely proportional to variance
            inv_var = 1.0 / (expert_unc + 1e-8)
            weights = inv_var / torch.sum(inv_var, dim=1, keepdim=True)
        else:
            # Standard Gating
            weights = self.gating(x).unsqueeze(-1)
            
        if self.prob_expert:
            return expert_out, expert_unc, weights
        return expert_out, None, weights