import torch
import torch.nn as nn
import torch.nn.functional as F
from models.PatchTST import Model as GatingModel
import copy

class Gating(nn.Module): 
    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Gating, self).__init__()
        self.projection = nn.Linear(1, configs.num_experts)
        configs = copy.deepcopy(configs)
        configs.prob_expert = 0
        self.gating_arc = GatingModel(configs)

    def gating(self, x_enc):
        enc_out = self.gating_arc.forecast(x_enc, None, None, None) # batch_size, seq_len, variate_dim
        enc_out = enc_out.unsqueeze(-1)
        weights = self.projection(enc_out)
        return weights # batch_size, seq_len, variate_dim, num_experts

    def forward(self, x_enc):
        weights = self.gating(x_enc)
        weights = F.softmax(weights, dim=-1)
        weights = weights.permute(0, 3, 1, 2)
        return weights



class Model(nn.Module):
    def __init__(self, configs, expert_model):
        super(Model, self).__init__()
        self.num_experts = configs.num_experts
        self.unc_gating = configs.unc_gating
        self.prob_expert = configs.prob_expert
        self.task_name = configs.task_name
        self.eps = 1e-09
        self.experts = nn.ModuleList([expert_model(configs).float() for _ in range(self.num_experts)])
        if not self.unc_gating:
            self.gating = Gating(configs).float()

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            expert_out = []
            expert_log_sigma = []
            for expert in self.experts:
                if self.prob_expert:
                    dec_out, log_sigma = expert(x_enc, x_mark_enc, 
                    x_dec, x_mark_dec, mask=None)
                    expert_log_sigma.append(log_sigma)
                else:
                    dec_out = expert.forward(x_enc, x_mark_enc,
                        x_dec, x_mark_dec, mask=None)
                expert_out.append(dec_out)
                
            expert_out = torch.stack(expert_out, dim=1) # [batch_size, num_experts, pred_len, num_features]
            if len(expert_log_sigma) > 0:
                expert_log_sigma = torch.stack(expert_log_sigma, dim=1) # [batch_size, num_experts, pred_len, num_features]
            if self.unc_gating:
                # Inverse variance weighting: higher confidence (lower uncertainty) gets higher weight
                expert_sq_sigma = torch.exp(expert_log_sigma)**2 + self.eps
                inv_var = 1.0 / (expert_sq_sigma+ 1e-8)  # [batch_size, num_experts, pred_len, num_features]
                sum_inv_var = torch.sum(inv_var, dim=1, keepdim=True)  # [batch_size, 1, pred_len, num_features]
                weights = inv_var / sum_inv_var  #  [batch_size, num_experts, pred_len, num_featuers]            else:
            else:    
                weights = self.gating(x_enc) # [batch_size, num_experts, pred_len, num_featuers]   
            return expert_out, expert_log_sigma, weights 
        else:
            raise NotImplementedError("{} not supported with MoE".format(self.task_name))
            