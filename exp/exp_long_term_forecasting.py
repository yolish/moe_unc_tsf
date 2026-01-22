from calibration.cp_vs_calibration import AdaptiveCPVS
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, visual_unc
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
from utils.losses import QuantileLoss
from calibration.cqr_calibration import OnlineCQRQuantile   
from calibration.cp_calibration import AdaptiveCP

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.max_grad_norm = 1

    def _build_model(self):
        if hasattr(self.args, 'use_quantile_loss') and self.args.use_quantile_loss:
            if self.args.c_out == self.args.enc_in:
                print(f"Force adjusting c_out from {self.args.c_out} to {self.args.c_out * 2} in build_model")
                self.args.c_out = self.args.c_out * 2

        base_model_cls = self.model_dict[self.args.model].Model

        class QuantileWrapper(nn.Module):
            def __init__(self, args):
                super().__init__()
                self.model = base_model_cls(args)
                self.projector = nn.Linear(args.enc_in, args.c_out)

            def forward(self, x, x_mark, dec_inp, y_mark, **kwargs):
                out = self.model(x, x_mark, dec_inp, y_mark, **kwargs)
                return self.projector(out)

        if self.args.use_quantile_loss and self.args.c_out > self.args.enc_in:
            print("Using QuantileWrapper to expand model output dimensions.")
            expert_model = QuantileWrapper
        else:
            expert_model = base_model_cls

        if self.args.moe:
            model = self.model_dict['MoE'].Model(self.args, expert_model).float()
        else:
            model = expert_model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
            if hasattr(self.args, 'use_quantile_loss') and self.args.use_quantile_loss:
                return QuantileLoss(quantiles=[0.05, 0.95])
                
            if self.args.moe:
                if self.args.prob_expert:
                    criterion = nn.GaussianNLLLoss(reduction='none')
                else:
                    criterion = nn.MSELoss(reduction='none')
            else:
                criterion = nn.MSELoss()
            return criterion
    
    def calc_aleatoric_epistermic_uncertainty(self, outputs, agg_outputs, 
                                              expert_unc, expert_weights):
        # Aleatoric uncertainty: weighted average of expert uncertainties
        aleatoric_unc = torch.sum(expert_unc * expert_weights, dim=1) #[batch_size, seq_len, num_feature]
        # Epistemic uncertainty: weighted variance of expert predictions
        epistemic_unc = None
        for i in range(self.args.num_experts):
            expert_diff = (agg_outputs - outputs[:, i, :, :])**2
            if epistemic_unc is None:
                epistemic_unc = expert_weights[:, i, :, :]*expert_diff
            else:
                epistemic_unc += expert_weights[:, i, :, :]*expert_diff
        return aleatoric_unc, epistemic_unc, aleatoric_unc+epistemic_unc
        
    
    def moe_loss(self, outputs, expert_unc, expert_weights, batch_y, criterion):
        # loss is a weighted sum of the loss of each expert per time step 
        loss = 0
        weighted_loss = None
        # expert_weights shape: [batch_size, num_experts, pred_len, num_features]
        for i in range(self.args.num_experts):
            expert_outputs = outputs[:, i, :, :]  # [batch_size, pred_len, num_features]
            if self.args.prob_expert: # guassian NLL
                expert_i_unc = expert_unc[:, i, :, :]
                expert_loss = criterion(expert_outputs,batch_y, expert_i_unc)
            else:
                expert_loss = criterion(expert_outputs,batch_y)
            expert_weight = expert_weights[:, i, :, :]  #  [batch_size, pred_len, num_features]
            # Element-wise multiplication and sum   
            if weighted_loss is None:
                weighted_loss = expert_loss * expert_weight
            else:
                weighted_loss += expert_loss * expert_weight
        
        loss = weighted_loss.mean()
        return loss  
 

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.moe:
                    outputs, expert_unc, expert_weights = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    
                    # MoE validation loss computation
                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = self.moe_loss(outputs, expert_unc, expert_weights, batch_y, criterion) 
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    pred = outputs.detach().cpu()
                    true = batch_y.detach().cpu()
                    loss = criterion(pred, true)

                total_loss.append(loss.cpu().numpy())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.moe:
                    outputs, expert_unc, expert_weights = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    
                    # MoE loss computation
                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    loss = self.moe_loss(outputs, expert_unc, expert_weights, batch_y, criterion) 
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                
                loss.backward()
                if self.args.prob_expert and self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                model_optim.step()
                

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
    
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        # only for prob. MoE
        epi_unc = []
        ale_unc = []
        weights = []
        per_expert_outputs = []
        per_expert_unc = []
        
        folder_path = './visual_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
       
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # encoder - decoder
                if self.args.moe:
                    outputs, expert_unc, expert_weights = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    if self.args.save_expert_outputs:
                        weights.append(expert_weights.detach().cpu().numpy())
                        per_expert_outputs.append(outputs.detach().cpu().numpy())
                        if self.args.prob_expert:
                            per_expert_unc.append(expert_unc.detach().cpu().numpy())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
                f_dim = -1 if self.args.features == 'MS' else 0
                
                outputs = outputs[:, -self.args.pred_len:, :]


                if hasattr(self.args, 'use_quantile_loss') and self.args.use_quantile_loss:
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device) 
                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()

                    n_feats = batch_y.shape[-1]
                    pred_low = outputs[:, :, :n_feats]
                    pred_high = outputs[:, :, n_feats:]

                    if test_data.scale and self.args.inverse:
                        shape = batch_y.shape
                        pred_low = test_data.inverse_transform(pred_low.reshape(shape[0] * shape[1], -1)).reshape(shape)
                        pred_high = test_data.inverse_transform(pred_high.reshape(shape[0] * shape[1], -1)).reshape(shape)
                        batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    
                    outputs = (pred_low + pred_high) / 2.0
                    
                else:
                    batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)           
                    outputs = outputs.detach().cpu().numpy()
                    batch_y = batch_y.detach().cpu().numpy()

                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                
                if self.args.moe:
                    outputs = torch.Tensor(outputs).to(self.device)
                    agg_outputs = torch.sum(outputs * expert_weights, dim=1) #[batch_size, seq_len, num_features]
                    if self.args.save_unc and self.args.prob_expert:
                        aleatoric_uncertainty, epistermic_uncertainty, _ = self.calc_aleatoric_epistermic_uncertainty(outputs, agg_outputs, 
                                                                            expert_unc, expert_weights)
                        
                        epi_unc.append(epistermic_uncertainty.cpu().numpy())
                        ale_unc.append(aleatoric_uncertainty.cpu().numpy())

                    outputs = agg_outputs.cpu().numpy() # [batch_size, seq_len, num_features]
                else:
                    outputs = outputs

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                    
                if self.args.save_visuals:
                    if i % 20 == 0:
                        input = batch_x.detach().cpu().numpy()
                        if test_data.scale and self.args.inverse:
                            shape = input.shape
                            input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                        gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                        visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        if self.args.save_outputs:
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)
        
        if len(weights) > 0:
           weights = np.concatenate(weights, axis=0)
           np.save(folder_path + "weights.npy", weights)
        if len(per_expert_outputs) > 0:
            per_expert_outputs = np.concatenate(per_expert_outputs, axis=0)
            np.save(folder_path + "per_expert_outputs.npy", per_expert_outputs)
        if len(per_expert_unc) > 0:
            per_expert_unc = np.concatenate(per_expert_unc, axis=0)
            np.save(folder_path + "per_expert_unc.npy", per_expert_unc)
        
        if len(epi_unc) > 0 :
            epi_unc = np.concatenate(epi_unc, axis=0)
            np.save(folder_path + 'epi_unc.npy', epi_unc)
            ale_unc = np.concatenate(ale_unc, axis=0)
            np.save(folder_path + 'ale_unc.npy', ale_unc)
        
        return

    def calibrate_cpvs(self, setting):
            print(">>>>>>> Start Calibration (CPVS) >>>>>>>>>>>")
            
            path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            if os.path.exists(path):
                self.model.load_state_dict(torch.load(path))
            self.model.eval()
            
            calibrator = AdaptiveCPVS(alpha=0.1, window_size=1000)

            def get_data_with_uncertainty(flag):
                data_set, loader = self._get_data(flag=flag) 
                preds_list, uncs_list, trues_list = [], [], []
                
                with torch.no_grad():
                    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
                        batch_x = batch_x.float().to(self.device)
                        batch_y = batch_y.float().to(self.device)
                        batch_x_mark = batch_x_mark.float().to(self.device)
                        batch_y_mark = batch_y_mark.float().to(self.device)
                        
                        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                        
                        if self.args.moe and self.args.prob_expert:
                            outputs, expert_unc, expert_weights = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            agg_outputs = torch.sum(outputs * expert_weights, dim=1)
                            _, _, total_variance = self.calc_aleatoric_epistermic_uncertainty(
                                outputs, agg_outputs, expert_unc, expert_weights
                            )
                            sigma = torch.sqrt(total_variance)
                            pred = agg_outputs
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            pred = outputs
                            sigma = torch.ones_like(pred) * 1e-6 

                        f_dim = -1 if self.args.features == 'MS' else 0
                        
                        preds_list.append(pred[:, -self.args.pred_len:, f_dim:].cpu().numpy())
                        uncs_list.append(sigma[:, -self.args.pred_len:, f_dim:].cpu().numpy())
                        trues_list.append(batch_y[:, -self.args.pred_len:, f_dim:].cpu().numpy())
                
                return np.concatenate(preds_list, axis=0), \
                    np.concatenate(uncs_list, axis=0), \
                    np.concatenate(trues_list, axis=0), \
                    data_set 

            val_preds, val_uncs, val_trues, _ = get_data_with_uncertainty('val')
            calibrator.fit(val_preds, val_uncs, val_trues)

            test_preds, test_uncs, test_trues, test_data_obj = get_data_with_uncertainty('test')
            
            final_lowers = []
            final_uppers = []
            q_history = [] 

            n_test = test_preds.shape[0]
            pred_len = self.args.pred_len
            
            last_q = None

            for t in range(n_test):
                window_changed = ((t - 1 - pred_len) >= 0)

                if last_q is None or window_changed:
                    lower, upper, curr_q = calibrator.predict_one_step(test_preds[t], test_uncs[t])
                    last_q = curr_q
                else:
                    curr_q = last_q
                    interval_width = curr_q * test_uncs[t]
                    lower = test_preds[t] - interval_width
                    upper = test_preds[t] + interval_width
                
                final_lowers.append(lower)
                final_uppers.append(upper)
                q_history.append(curr_q)
                
                t_update = t - pred_len
                if t_update >= 0:
                    calibrator.update(test_preds[t_update], test_uncs[t_update], test_trues[t_update])

            final_lowers = np.array(final_lowers)
            final_uppers = np.array(final_uppers)

            if test_data_obj.scale and self.args.inverse:
                print("Applying Inverse Transform to metrics...")
                shape = final_lowers.shape
                final_lowers = test_data_obj.inverse_transform(final_lowers.reshape(shape[0] * shape[1], -1)).reshape(shape)
                final_uppers = test_data_obj.inverse_transform(final_uppers.reshape(shape[0] * shape[1], -1)).reshape(shape)
                test_trues = test_data_obj.inverse_transform(test_trues.reshape(shape[0] * shape[1], -1)).reshape(shape)

            coverage = np.mean((test_trues >= final_lowers) & (test_trues <= final_uppers))
            width = np.mean(np.abs(final_uppers - final_lowers))
            
            print(f"\nAdaptive CPVS Results (Delayed):")
            print(f"Mean q: {np.mean(q_history):.4f}")
            print(f"Coverage: {coverage:.4f}")
            print(f"Avg Width: {width:.4f}")
            
            folder_path = './results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                
            with open("result_calibration_cpvs.txt", 'a') as f:
                f.write(f"{setting} (Adaptive CPVS Delayed)\n")
                f.write(f"q_mean: {np.mean(q_history):.4f}, Coverage: {coverage:.4f}, Width: {width:.4f}\n\n")
                
            return coverage, width
        
    def calibrate_cqr(self, setting):
        path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
        self.model.eval()
        
        calibrator = OnlineCQRQuantile(alpha=0.1, window_size=1000)

        def get_quantile_preds(flag):
            data_set, loader = self._get_data(flag=flag) 
            lowers, uppers, trues = [], [], []
            
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                    
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    
                    f_dim = -1 if self.args.features == 'MS' else 0
                    true_y = batch_y[:, -self.args.pred_len:, f_dim:]
                    
                    n_feats = true_y.shape[-1]
                    pred_lower = outputs[:, -self.args.pred_len:, :n_feats]      
                    pred_upper = outputs[:, -self.args.pred_len:, n_feats:]    

                    lowers.append(pred_lower.cpu().numpy())
                    uppers.append(pred_upper.cpu().numpy())
                    trues.append(true_y.cpu().numpy())
            
            return np.concatenate(lowers, 0), np.concatenate(uppers, 0), np.concatenate(trues, 0), data_set

        val_low, val_high, val_true, _ = get_quantile_preds('val')
        calibrator.fit(val_low, val_high, val_true)

        test_low, test_high, test_true, test_data_obj = get_quantile_preds('test')
        
        final_lowers, final_uppers = [], []
        q_history = []
        
        pred_len = self.args.pred_len
        last_q = None

        print(f"Starting Sliding Window CQR with Delay of {pred_len} steps...")

        for t in range(len(test_true)):
            window_changed = ((t - 1 - pred_len) >= 0)

            if last_q is None or window_changed:
                l, u, q = calibrator.predict_one_step(test_low[t], test_high[t])
                last_q = q
            else:
                q = last_q
                l = test_low[t] - q
                u = test_high[t] + q

            final_lowers.append(l)
            final_uppers.append(u)
            q_history.append(q)
            
            t_update = t - pred_len
            if t_update >= 0:
                calibrator.update(test_low[t_update], test_high[t_update], test_true[t_update])
            
        final_lowers = np.array(final_lowers)
        final_uppers = np.array(final_uppers)

        if test_data_obj.scale and self.args.inverse:
            print("Applying Inverse Transform to CQR results...")
            shape = final_lowers.shape
            final_lowers = test_data_obj.inverse_transform(final_lowers.reshape(shape[0] * shape[1], -1)).reshape(shape)
            final_uppers = test_data_obj.inverse_transform(final_uppers.reshape(shape[0] * shape[1], -1)).reshape(shape)
            test_true = test_data_obj.inverse_transform(test_true.reshape(shape[0] * shape[1], -1)).reshape(shape)

        coverage = np.mean((test_true >= final_lowers) & (test_true <= final_uppers))
        width = np.mean(np.abs(final_uppers - final_lowers))
        
        print(f"Coverage: {coverage:.4f}, Width: {width:.4f}")
        
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        with open("result_calibration_cqr_quantile.txt", 'a') as f:
            f.write(f"{setting} (CQR Quantile alpha=0.1, Delayed Update)\n")
            f.write(f"q_mean: {np.mean(q_history):.4f}, Coverage: {coverage:.4f}, Width: {width:.4f}\n\n")
            
        return coverage, width
    
    def calibrate_cp(self, setting):
        print(">>>>>>> Start Standard CP Calibration (Sliding Window) >>>>>>>>>>>")
        
        # Load best model
        path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
        self.model.eval()
        
        calibrator = AdaptiveCP(alpha=0.1, window_size=1000)

        def get_deterministic_preds(flag):
            data_set, loader = self._get_data(flag=flag) 
            preds_list, trues_list = [], []
            
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                    
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    
                    if self.args.moe:
                        outputs, _, expert_weights = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        pred = torch.sum(outputs * expert_weights, dim=1) 
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        pred = outputs

                    f_dim = -1 if self.args.features == 'MS' else 0
                    
                    preds_list.append(pred[:, -self.args.pred_len:, f_dim:].cpu().numpy())
                    trues_list.append(batch_y[:, -self.args.pred_len:, f_dim:].cpu().numpy())
            
            return np.concatenate(preds_list, axis=0), \
                   np.concatenate(trues_list, axis=0), \
                   data_set 

        print("Fitting calibrator on Validation set...")
        val_preds, val_trues, _ = get_deterministic_preds('val')
        calibrator.fit(val_preds, val_trues)

        print("Running Online Calibration on Test set...")
        test_preds, test_trues, test_data_obj = get_deterministic_preds('test')
        
        final_lowers = []
        final_uppers = []
        q_history = [] 

        n_test = test_preds.shape[0]
        pred_len = self.args.pred_len
        last_q = None

        for t in range(n_test):

            window_changed = ((t - 1 - pred_len) >= 0)

            if last_q is None or window_changed:
                lower, upper, curr_q = calibrator.predict_one_step(test_preds[t])
                last_q = curr_q
            else:
                curr_q = last_q
                interval_width = curr_q 
                lower = test_preds[t] - interval_width
                upper = test_preds[t] + interval_width
            
            final_lowers.append(lower)
            final_uppers.append(upper)
            q_history.append(curr_q)
            
            t_update = t - pred_len
            if t_update >= 0:
                calibrator.update(test_preds[t_update], test_trues[t_update])

        final_lowers = np.array(final_lowers)
        final_uppers = np.array(final_uppers)

        if test_data_obj.scale and self.args.inverse:
            print("Applying Inverse Transform to metrics...")
            shape = final_lowers.shape
            final_lowers = test_data_obj.inverse_transform(final_lowers.reshape(shape[0] * shape[1], -1)).reshape(shape)
            final_uppers = test_data_obj.inverse_transform(final_uppers.reshape(shape[0] * shape[1], -1)).reshape(shape)
            test_trues = test_data_obj.inverse_transform(test_trues.reshape(shape[0] * shape[1], -1)).reshape(shape)

        coverage = np.mean((test_trues >= final_lowers) & (test_trues <= final_uppers))
        width = np.mean(np.abs(final_uppers - final_lowers))
        
        print(f"\nStandard CP Results:")
        print(f"Mean q (Absolute Error Quantile): {np.mean(q_history):.4f}")
        print(f"Coverage: {coverage:.4f}")
        print(f"Avg Width: {width:.4f}")
        
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        with open("result_calibration_mse_cp.txt", 'a') as f:
            f.write(f"{setting} (Standard CP with Sliding Window)\n")
            f.write(f"q_mean: {np.mean(q_history):.4f}, Coverage: {coverage:.4f}, Width: {width:.4f}\n\n")
            
        return coverage, width