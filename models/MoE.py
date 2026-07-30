import torch
import torch.nn as nn
import torch.nn.functional as F
from models.PatchTST import Model as GatingModel
import copy

class Gating(nn.Module):
    def __init__(self, configs, individual=False):
        super(Gating, self).__init__()
        self.projection = nn.Linear(1, configs.num_experts)
        configs = copy.deepcopy(configs)
        configs.prob_expert = 0
        self.gating_arc = GatingModel(configs)

    def gating(self, x_enc):
        enc_out = self.gating_arc.forecast(x_enc, None, None, None) 
        enc_out = enc_out.unsqueeze(-1)
        weights = self.projection(enc_out)
        return weights 

    def forward(self, x_enc):
        weights = self.gating(x_enc)
        weights = F.softmax(weights, dim=-1)
        weights = weights.permute(0, 3, 1, 2)
        return weights

class Model(nn.Module):
    def __init__(self, configs, expert_model=None):
        super(Model, self).__init__()
        self.num_experts = configs.num_experts
        self.unc_gating = configs.unc_gating
        self.prob_expert = configs.prob_expert
        self.task_name = configs.task_name

        self.experts = nn.ModuleList([expert_model(configs).float() for _ in range(self.num_experts)])
        if not self.unc_gating:
            self.gating = Gating(configs).float()

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == 'long_term_forecast':
            expert_out = []
            expert_unc = []
            for expert in self.experts:
                sq_sigma = None
                if self.prob_expert:
                    dec_out, sq_sigma = expert(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)
                    expert_unc.append(sq_sigma)
                else:
                    dec_out = expert.forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)
                expert_out.append(dec_out)
                
            expert_out = torch.stack(expert_out, dim=1) 
            if len(expert_unc) > 0:
                expert_unc = torch.stack(expert_unc, dim=1) 
            if self.unc_gating:
                inv_var = 1.0 / (expert_unc + 1e-8)  
                sum_inv_var = torch.sum(inv_var, dim=1, keepdim=True)  
                weights = inv_var / sum_inv_var  
            else:    
                weights = self.gating(x_enc)   
            return expert_out, expert_unc, weights 
        else:
            raise NotImplementedError("{} not supported with MoE".format(self.task_name))

