import torch
import torch.nn as nn
import torch.nn.functional as F
from models.PatchTST import Model as GatingModel
import copy

class MLP_Expert(nn.Module):
    def __init__(self, configs):
        super(MLP_Expert, self).__init__()
        self.prob_expert = configs.prob_expert
        hidden_dim = configs.d_model if hasattr(configs, 'd_model') else 64
        dropout_rate = configs.dropout if hasattr(configs, 'dropout') else 0.1
        
        # בניית 3 שכבות חבויות בגודל זהה עם ReLU ו-Dropout לפי המאמר [cite: 581, 583]
        self.feature_extractor = nn.Sequential(
            nn.Linear(configs.enc_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # שכבת הפלט מחוברת ישירות לשכבה החבויה השלישית
        self.mean_head = nn.Linear(hidden_dim, configs.c_out)
        
        if self.prob_expert:
            self.var_head = nn.Sequential(
                nn.Linear(hidden_dim, configs.c_out),
                nn.Softplus()
            )

    def forward(self, x):
        features = self.feature_extractor(x)
        mean = self.mean_head(features)
        
        if self.prob_expert:
            var = self.var_head(features)
            return mean, var
        return mean

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
        
        if self.task_name == 'tabular_regression':
            self.experts = nn.ModuleList([MLP_Expert(configs).float() for _ in range(self.num_experts)])
            if not self.unc_gating:
                hidden_dim = configs.d_model if hasattr(configs, 'd_model') else 64
                dropout_rate = configs.dropout if hasattr(configs, 'dropout') else 0.1
                
                # התאמת הנתב למבנה של 3 שכבות חבויות זהות למומחה 
                self.tabular_router = nn.Sequential(
                    nn.Linear(configs.enc_in, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    
                    nn.Linear(hidden_dim, self.num_experts)
                )
        else:
            self.experts = nn.ModuleList([expert_model(configs).float() for _ in range(self.num_experts)])
            if not self.unc_gating:
                self.gating = Gating(configs).float()

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == 'tabular_regression':
            expert_outputs = []
            expert_unc = []
            
            for expert in self.experts:
                if self.prob_expert:
                    mean, var = expert(x_enc)
                    expert_outputs.append(mean)
                    expert_unc.append(var)
                else:
                    expert_outputs.append(expert(x_enc))
            
            expert_outputs = torch.stack(expert_outputs, dim=1)
            
            if self.unc_gating and self.prob_expert:
                expert_unc = torch.stack(expert_unc, dim=1)
                inv_var = 1.0 / (expert_unc + 1e-8)
                sum_inv_var = torch.sum(inv_var, dim=1, keepdim=True)
                gating_weights = (inv_var / sum_inv_var).squeeze(-1)
                output = torch.sum(gating_weights.unsqueeze(-1) * expert_outputs, dim=1)
            else:
                gating_logits = self.tabular_router(x_enc)
                gating_weights = F.softmax(gating_logits, dim=-1)
                output = torch.sum(gating_weights.unsqueeze(-1) * expert_outputs, dim=1)
                
                if self.prob_expert:
                    expert_unc = torch.stack(expert_unc, dim=1)
            
            if self.prob_expert:
                # 1. Aleatoric variance (data noise - expected value of variances)
                aleatoric_var = torch.sum(gating_weights.unsqueeze(-1) * expert_unc, dim=1)
                # 2. Epistemic variance (model uncertainty - variance of expected values)
                epistemic_var = torch.sum(gating_weights.unsqueeze(-1) * (expert_outputs - output.unsqueeze(1))**2, dim=1)
                # 3. Total variance
                total_std = torch.sqrt(aleatoric_var + epistemic_var).squeeze(-1)
                aleatoric_std = torch.sqrt(aleatoric_var).squeeze(-1)
                epistemic_std = torch.sqrt(epistemic_var).squeeze(-1)
            else:
                total_std = torch.std(expert_outputs, dim=1).squeeze(-1) if self.num_experts > 1 else torch.ones_like(output).squeeze(-1)
                aleatoric_std = total_std
                epistemic_std = total_std 
                expert_unc = None
                
            return output, gating_weights, total_std, aleatoric_std, epistemic_std, expert_outputs, expert_unc

        elif self.task_name == 'long_term_forecast':
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