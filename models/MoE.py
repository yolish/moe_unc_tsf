import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp


class Gating(nn.Module): # based on DLinear model 
    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(Gating, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        # Series decomposition block from Autoformer
        self.decompsition = series_decomp(configs.moving_avg)
        self.individual = individual
        self.channels = configs.enc_in
        self.pred_len = configs.pred_len
        self.num_experts = configs.num_experts

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(
                    nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(
                    nn.Linear(self.seq_len, self.pred_len))

                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        
        self.projection = nn.Linear(
            configs.enc_in * configs.seq_len, configs.num_experts*configs.pred_len)

    def encoder(self, x):
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)

    def gating(self, x_enc):
        # Encoder
        enc_out = self.encoder(x_enc)
        # Output
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        # (batch_size, num_classes)
        output = self.projection(output)
        return output

    def forward(self, x_enc):
        dec_out = self.gating(x_enc)
        # reshape and pass through softmax to give weights per expert 
        weights = dec_out.reshape(dec_out.shape[0], self.num_experts, self.pred_len)
        weights = F.softmax(dec_out, dim=1)
        return weights



class Model(nn.Module):
    def __init__(self, configs, expert_model):
        super(Model, self).__init__()
        self.num_experts = configs.num_experts
        self.unc_gating = configs.unc_gating
        self.prob_expert = configs.prob_expert
        self.task_name = configs.task_name
        self.experts = nn.ModuleList([expert_model.Model(configs).float() for _ in range(self.num_experts)])
        if not self.unc_gating:
            self.gating = Gating(configs).float()

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            expert_out = []
            expert_unc = []
            for expert in self.experts:
                sq_sigma = None
                if self.prob_expert:
                    dec_out, log_sq_sigma_out = expert.formard(x_enc, x_mark_enc, 
                    x_dec, x_mark_dec, mask=None)
                    sq_sigma = torch.exp(log_sq_sigma_out)
                else:
                    dec_out, _ = expert.forward(x_enc, x_mark_enc,
                     x_dec, x_mark_dec, mask=None)
                expert_out.append(dec_out)
                expert_unc.append(sq_sigma)
            if self.unc_gating:
                # Stack expert uncertainties and compute weights based on inverse variance
                expert_unc_stacked = torch.stack(expert_unc, dim=0)  # [num_experts, batch_size, seq_len, features]
                # Average across features to get per-expert uncertainty
                expert_unc_avg = torch.mean(expert_unc_stacked, dim=-1)  # [num_experts, batch_size, seq_len]
                # Inverse variance weighting: higher confidence (lower uncertainty) gets higher weight
                inv_var = 1.0 / (expert_unc_avg + 1e-8)  # [num_experts, batch_size, seq_len]
                sum_inv_var = torch.sum(inv_var, dim=0, keepdim=True)  # [1, batch_size, seq_len]
                weights_temp = inv_var / sum_inv_var  # [num_experts, batch_size, seq_len]
                weights = weights_temp.permute(1, 0, 2)  # [batch_size, num_experts, seq_len]
            else:
                weights = self.gating(x_enc)
            
            return expert_out, expert_unc, weights 
        else:
            raise NotImplementedError("{} not supported with MoE".format(self.task_name))
            