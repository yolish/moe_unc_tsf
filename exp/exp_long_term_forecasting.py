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
from calibration.aci_aleatoric_scale_calibration import ACIAleatoricScaleCalibrator
from calibration.aci_cp_calibration import ACICPCalibrator
from calibration.aci_cpvs_calibration import ACICPVSCalibrator
from calibration.aci_cqr_calibration import ACICQRCalibrator
from data_provider.data_loader import Dataset_Window
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')


# --------------------------------------------------------------------------- MoECP
# MoECP recomputes an H x C grid of weighted quantiles at every forecast origin, and the
# per-origin cost is O(W log W * H * C). On the 7-channel ETT that is minutes; on traffic
# (862 channels) it is ~13 s/origin, i.e. half a day. Every reduction inside
# `localized_quantile` runs along the window axis and the einsum contracts only the expert
# axis, so the channel axis is separable and each block can run in its own process.
#
# The one thing that is NOT separable is the multinomial randomization: `calibrator.rng` is
# a single stream consumed row-major over the (h, c) grid, so a worker re-seeding its own
# Generator would draw a different pi~ and silently produce a different -- not wrong, but
# different -- realization. The parent therefore keeps sole ownership of the RNG, draws the
# full [H, C, K] pi~ in serial order, and scatters channel slices. Workers touch no RNG, so
# the result is bit-identical to the serial path. Drawing pi~ is ~0.3% of a step, so
# centralizing it costs nothing.

def _moecp_worker(conn, calibrator, preds, gates, trues, pred_len, cols):
    """Run the full origin loop for one contiguous block of channels.

    Inherited by fork, so `preds`/`gates`/`trues` are copy-on-write rather than pickled;
    only the per-origin pi~ slice crosses the pipe.
    """
    try:
        calibrator.resid_window = np.ascontiguousarray(calibrator.resid_window[:, :, cols])
        calibrator.loggate_window = np.ascontiguousarray(calibrator.loggate_window[:, :, cols])
        p = np.ascontiguousarray(preds[:, :, cols])
        g = np.ascontiguousarray(gates[:, :, cols])
        y = np.ascontiguousarray(trues[:, :, cols])

        for t in range(p.shape[0]):
            pi_t = conn.recv()
            if pi_t is StopIteration:
                return
            _, _, q = calibrator.predict_one_step(p[t], g[t], pi_tilde=pi_t)
            conn.send(q)
            t_update = t - pred_len
            if t_update >= 0:
                calibrator.update(p[t_update], g[t_update], y[t_update])
        conn.send(None)
    except Exception as exc:                     # surface the traceback to the parent
        import traceback
        conn.send(('__error__', traceback.format_exc(), repr(exc)))
    finally:
        conn.close()


def _moecp_q_parallel(calibrator, test_preds, test_gates, test_trues, pred_len, workers):
    """Channel-parallel MoECP. Returns the [N, H, C] quantile grid the serial loop builds."""
    import multiprocessing as mp

    n_test, _, n_chan = test_preds.shape
    blocks = [b for b in np.array_split(np.arange(n_chan), min(workers, n_chan)) if b.size]
    ctx = mp.get_context('fork')

    procs, pipes = [], []
    for cols in blocks:
        parent_conn, child_conn = ctx.Pipe()
        pr = ctx.Process(target=_moecp_worker,
                         args=(child_conn, calibrator, test_preds, test_gates, test_trues,
                               pred_len, cols),
                         daemon=True)
        pr.start()
        child_conn.close()
        procs.append(pr)
        pipes.append(parent_conn)

    print(f"MoECP: {len(blocks)} channel-parallel workers over {n_chan} channels, "
          f"{n_test} origins")
    q_all = np.empty_like(test_preds)
    t_start = time.time()
    try:
        for t in range(n_test):
            # Mirror the serial recompute schedule exactly, so the RNG stream advances on
            # the same steps it would have serially.
            pi_t = (calibrator._randomized_gate(test_gates[t])
                    if calibrator.needs_recompute() else None)
            calibrator._last_q = True            # only the schedule matters in the parent
            calibrator._steps += 1

            for conn, cols in zip(pipes, blocks):
                conn.send(pi_t[:, cols] if pi_t is not None else None)
            for conn, cols in zip(pipes, blocks):
                out = conn.recv()
                if isinstance(out, tuple) and out and out[0] == '__error__':
                    raise RuntimeError(f"MoECP worker failed:\n{out[1]}")
                q_all[t][:, cols] = out

            if t and t % 250 == 0:
                rate = (time.time() - t_start) / t
                print(f"  origin {t}/{n_test}  {rate:.3f} s/origin  "
                      f"eta {rate * (n_test - t) / 60:.1f} min", flush=True)
    finally:
        for conn in pipes:
            try:
                conn.send(StopIteration)
                conn.close()
            except (BrokenPipeError, OSError):
                pass
        for pr in procs:
            pr.join(timeout=30)
            if pr.is_alive():
                pr.terminate()
    return q_all


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
        
    
    def _moe_aggregate(self, outputs, expert_weights):
        """Collapse the per-expert dimension of a MoE forward into a single prediction.

        Under --use_quantile_loss the experts emit 2*enc_in channels ([lower | upper]),
        while the gating head (PatchTST, channel-independent) emits enc_in weights - one
        per feature. Tile those weights across both halves so a feature's mixture weight
        applies to its lower and upper quantile alike.
        """
        if expert_weights.shape[-1] != outputs.shape[-1]:
            reps = outputs.shape[-1] // expert_weights.shape[-1]
            if reps * expert_weights.shape[-1] != outputs.shape[-1]:
                raise ValueError(
                    f"Cannot align gating weights {tuple(expert_weights.shape)} with expert "
                    f"outputs {tuple(outputs.shape)}: channel count is not an integer multiple.")
            expert_weights = torch.cat([expert_weights] * reps, dim=-1)
        return torch.sum(outputs * expert_weights, dim=1)

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
                    if hasattr(self.args, 'use_quantile_loss') and self.args.use_quantile_loss:
                        # Collapse the per-expert dimension now: the quantile split below
                        # assumes a 3D (batch, seq, 2*n_feats) tensor, same as the non-MoE path.
                        outputs = self._moe_aggregate(outputs, expert_weights)
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

                
                if self.args.moe and not (hasattr(self.args, 'use_quantile_loss') and self.args.use_quantile_loss):
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
                    
                    if self.args.moe:
                        # MoE returns (expert_out, expert_unc, weights); aggregate the
                        # per-expert quantile predictions the same way calibrate_cpvs does.
                        outputs, _, expert_weights = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs = self._moe_aggregate(outputs, expert_weights)
                    else:
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

    # ------------------------------------------------------------------
    # Aleatoric/epistemic-separated online CP (CP-VS family)
    # ------------------------------------------------------------------

    def _collect_separated_uncertainty(self, flag):
        """Run the model over a split, keeping aleatoric and epistemic variance apart.

        Mirrors the collection loop in calibrate_cpvs, except that the two variance
        components are returned separately instead of being summed into a total sigma.
        """
        data_set, loader = self._get_data(flag=flag)
        preds_list, ale_list, epi_list, trues_list = [], [], [], []

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
                    agg_outputs = self._moe_aggregate(outputs, expert_weights)
                    ale_unc, epi_unc, _ = self.calc_aleatoric_epistermic_uncertainty(
                        outputs, agg_outputs, expert_unc, expert_weights
                    )
                    pred = agg_outputs
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    pred = outputs
                    ale_unc = torch.ones_like(pred) * 1e-6
                    epi_unc = torch.zeros_like(pred)

                f_dim = -1 if self.args.features == 'MS' else 0

                preds_list.append(pred[:, -self.args.pred_len:, f_dim:].cpu().numpy())
                ale_list.append(ale_unc[:, -self.args.pred_len:, f_dim:].cpu().numpy())
                epi_list.append(epi_unc[:, -self.args.pred_len:, f_dim:].cpu().numpy())
                trues_list.append(batch_y[:, -self.args.pred_len:, f_dim:].cpu().numpy())

        return np.concatenate(preds_list, axis=0), \
            np.concatenate(ale_list, axis=0), \
            np.concatenate(epi_list, axis=0), \
            np.concatenate(trues_list, axis=0), \
            data_set

    def _run_separated_calibration(self, setting, calibrator, use_epistemic, title, result_file,
                                   extra_fields=None):
        """Delayed-update online calibration loop shared by the aleatoric CP-VS variants.

        Same protocol as calibrate_cpvs: q is refreshed only once the score window has
        rolled past the horizon (so no target is used before it would be observable),
        and each window is fed back pred_len steps late.

        `extra_fields` is an optional zero-arg callable, evaluated after fit(), returning
        a string appended to the metrics line -- for calibrators that learn a parameter
        worth recording alongside coverage and width.
        """
        path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
        self.model.eval()

        val_preds, val_ale, val_epi, val_trues, _ = self._collect_separated_uncertainty('val')
        test_preds, test_ale, test_epi, test_trues, test_data_obj = self._collect_separated_uncertainty('test')

        def unc_at(t):
            return (test_ale[t], test_epi[t]) if use_epistemic else (test_ale[t],)

        if use_epistemic:
            calibrator.fit(val_preds, val_ale, val_epi, val_trues)
        else:
            calibrator.fit(val_preds, val_ale, val_trues)

        final_lowers, final_uppers, q_history = [], [], []
        n_test = test_preds.shape[0]
        pred_len = self.args.pred_len
        last_q = None

        for t in range(n_test):
            window_changed = ((t - 1 - pred_len) >= 0)

            if last_q is None or window_changed:
                lower, upper, curr_q = calibrator.predict_one_step(test_preds[t], *unc_at(t))
                last_q = curr_q
            else:
                curr_q = last_q
                lower, upper = calibrator.interval_from_q(curr_q, test_preds[t], *unc_at(t))

            final_lowers.append(lower)
            final_uppers.append(upper)
            q_history.append(np.mean(curr_q))

            t_update = t - pred_len
            if t_update >= 0:
                calibrator.update(test_preds[t_update], *unc_at(t_update), test_trues[t_update])

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

        print(f"\n{title} Results (Delayed):")
        print(f"Mean q: {np.mean(q_history):.4f}")
        print(f"Coverage: {coverage:.4f}")
        print(f"Avg Width: {width:.4f}")

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        extra = extra_fields() if extra_fields is not None else ""

        with open(result_file, 'a') as f:
            f.write(f"{setting} ({title} Delayed)\n")
            f.write(f"q_mean: {np.mean(q_history):.4f}, Coverage: {coverage:.4f}, "
                    f"Width: {width:.4f}{extra}\n\n")

        return coverage, width

    def calibrate_aci_aleatoric_scale(self, setting):
        print(">>>>>>> Start Calibration (ACI Aleatoric Scale) >>>>>>>>>>>")
        calibrator = ACIAleatoricScaleCalibrator(
            alpha=self.args.aci_alpha, gamma=self.args.aci_gamma, window_size=1000)
        coverage, width = self._run_separated_calibration(
            setting,
            calibrator,
            use_epistemic=True,
            title="ACI Aleatoric Scale CP",
            result_file="result_calibration_aci_aleatoric_scale_tsf.txt",
            extra_fields=lambda: (f", gamma: {calibrator.gamma}"
                                  f", alpha_t_mean: {np.mean(calibrator.alpha_t_history):.4f}"
                                  f", alpha_t_final: {np.mean(calibrator.alpha_t):.4f}"
                                  f", alpha_t_p05: {np.percentile(calibrator.alpha_t, 5):.4f}"
                                  f", alpha_t_p95: {np.percentile(calibrator.alpha_t, 95):.4f}"
                                  f", err_rate: {calibrator.err_rate_:.4f}"))
        # alpha_t is clipped to [0, 1], so p05 near 0 flags cells parked at the window-max
        # rather than tracking their own miscoverage -- the windup the clip trades in for a
        # bounded interval. See the calibrator's docstring for the full trade.
        print(f"Final alpha_t = {np.mean(calibrator.alpha_t):.4f} mean, "
              f"[{np.percentile(calibrator.alpha_t, 5):.4f}, "
              f"{np.percentile(calibrator.alpha_t, 95):.4f}] p05-p95 "
              f"(target {calibrator.alpha}), realized miscoverage = {calibrator.err_rate_:.4f}")
        return coverage, width

    def calibrate_aci_aleatoric_scale_g001(self, setting):
        """Same calibrator as calibrate_aci_aleatoric_scale, gamma fixed at 0.001.

        A slower step keeps alpha_t off the [0, 1] clip more often (the loop-gain scale is
        ~gamma*pred_len under the delayed-update protocol), trading responsiveness to drift
        for less windup. Kept as a separate flag/result-file, not a parameter sweep, so both
        gammas can be run and compared side by side rather than one replacing the other.
        """
        print(">>>>>>> Start Calibration (ACI Aleatoric Scale, g=0.001) >>>>>>>>>>>")
        calibrator = ACIAleatoricScaleCalibrator(
            alpha=self.args.aci_alpha, gamma=0.001, window_size=1000)
        coverage, width = self._run_separated_calibration(
            setting,
            calibrator,
            use_epistemic=True,
            title="ACI Aleatoric Scale CP (g=0.001)",
            result_file="result_calibration_aci_aleatoric_scale_g001_tsf.txt",
            extra_fields=lambda: (f", gamma: {calibrator.gamma}"
                                  f", alpha_t_mean: {np.mean(calibrator.alpha_t_history):.4f}"
                                  f", alpha_t_final: {np.mean(calibrator.alpha_t):.4f}"
                                  f", alpha_t_p05: {np.percentile(calibrator.alpha_t, 5):.4f}"
                                  f", alpha_t_p95: {np.percentile(calibrator.alpha_t, 95):.4f}"
                                  f", err_rate: {calibrator.err_rate_:.4f}"))
        print(f"Final alpha_t = {np.mean(calibrator.alpha_t):.4f} mean, "
              f"[{np.percentile(calibrator.alpha_t, 5):.4f}, "
              f"{np.percentile(calibrator.alpha_t, 95):.4f}] p05-p95 "
              f"(target {calibrator.alpha}), realized miscoverage = {calibrator.err_rate_:.4f}")
        return coverage, width

    def _collect_gating(self, flag):
        """Run the model over a split, returning predictions, gate distributions and targets.

        Same collection loop as `_collect_separated_uncertainty`, but it keeps the gating
        weights pi_k (moved to a trailing expert axis) instead of reducing them into
        variance components -- MoECP localizes on the gate itself.

        Returns
        -------
        preds [N, H, C], gates [N, H, C, K], trues [N, H, C], data_set
        """
        data_set, loader = self._get_data(flag=flag)
        preds_list, gates_list, trues_list = [], [], []

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs, expert_unc, expert_weights = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = self._moe_aggregate(outputs, expert_weights)

                f_dim = -1 if self.args.features == 'MS' else 0
                # [B, K, L, C] -> [B, L, C, K]
                gates = expert_weights[:, :, -self.args.pred_len:, f_dim:].permute(0, 2, 3, 1)

                preds_list.append(pred[:, -self.args.pred_len:, f_dim:].cpu().numpy())
                gates_list.append(gates.cpu().numpy().astype(np.float32))
                trues_list.append(batch_y[:, -self.args.pred_len:, f_dim:].cpu().numpy())

        return np.concatenate(preds_list, axis=0), \
            np.concatenate(gates_list, axis=0), \
            np.concatenate(trues_list, axis=0), \
            data_set

    def _build_full_series(self):
        """Reconstruct the transformed full series from the three standard splits.

        The Dataset_* classes always fit the scaler on the train slice regardless of
        set_type, so all three splits share an identical `data` array, and each
        non-train split is back-shifted by exactly seq_len. Concatenating them while
        dropping those overlaps rebuilds data[0:border2s[2]] with the original
        (frozen) scaler, without touching the hard-coded border logic.
        """
        seq_len = self.args.seq_len

        train_set, _ = self._get_data(flag='train')
        val_set, _ = self._get_data(flag='val')
        test_set, _ = self._get_data(flag='test')

        full_x = np.concatenate(
            [train_set.data_x, val_set.data_x[seq_len:], test_set.data_x[seq_len:]], axis=0)
        full_stamp = np.concatenate(
            [train_set.data_stamp, val_set.data_stamp[seq_len:], test_set.data_stamp[seq_len:]], axis=0)

        test_offset = len(full_x) - len(test_set.data_x)

        return full_x, full_stamp, test_offset, val_set, test_set

    def _window_loader(self, full_x, full_stamp, raw_start, raw_end, shuffle=False):
        size = [self.args.seq_len, self.args.label_len, self.args.pred_len]
        data_set = Dataset_Window(full_x[raw_start:raw_end], full_stamp[raw_start:raw_end], size)
        loader = DataLoader(data_set, batch_size=self.args.batch_size, shuffle=shuffle,
                            num_workers=self.args.num_workers, drop_last=False)
        return data_set, loader

    def _predict_quantiles(self, loader):
        """Batched quantile inference; mirrors get_quantile_preds() in calibrate_cqr."""
        self.model.eval()
        lowers, uppers, trues = [], [], []

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
                    outputs = self._moe_aggregate(outputs, expert_weights)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                true_y = batch_y[:, -self.args.pred_len:, f_dim:]

                n_feats = true_y.shape[-1]
                pred_lower = outputs[:, -self.args.pred_len:, :n_feats]
                pred_upper = outputs[:, -self.args.pred_len:, n_feats:]

                lowers.append(pred_lower.cpu().numpy())
                uppers.append(pred_upper.cpu().numpy())
                trues.append(true_y.cpu().numpy())

        return np.concatenate(lowers, 0), np.concatenate(uppers, 0), np.concatenate(trues, 0)

    def _retrain_on_window(self, full_x, full_stamp, raw_start, raw_end):
        """Re-fit the model on a trailing window (ACI-style rolling re-estimation).

        Deliberately does NOT use EarlyStopping and never writes a checkpoint, so
        checkpoints/<setting>/checkpoint.pth stays intact. adjust_learning_rate is
        skipped too: the default 'type1' schedule halves the LR every epoch, which
        would strand a freshly re-initialized model at lr/2^epochs.
        """
        mode = getattr(self.args, 'retrain_mode', 'finetune')
        epochs = getattr(self.args, 'retrain_epochs', 3)
        lr = getattr(self.args, 'retrain_lr', None) or self.args.learning_rate

        if mode == 'scratch':
            self.model = self._build_model().to(self.device)

        _, loader = self._window_loader(full_x, full_stamp, raw_start, raw_end, shuffle=True)

        model_optim = optim.Adam(self.model.parameters(), lr=lr)
        criterion = self._select_criterion()

        self.model.train()
        for epoch in range(epochs):
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y_t = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                if self.args.moe:
                    outputs, expert_unc, expert_weights = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    loss = self.moe_loss(outputs, expert_unc, expert_weights, batch_y_t, criterion)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    loss = criterion(outputs, batch_y_t)

                loss.backward()
                if self.args.prob_expert and self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                model_optim.step()

        self.model.eval()

    def calibrate_cqr_retrain(self, setting):
        print(">>>>>>> Start Retrained CQR Calibration (Rolling Window) >>>>>>>>>>>")

        path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
        self.model.eval()

        assert getattr(self.args, 'augmentation_ratio', 0) == 0, \
            "calibrate_cqr_retrain requires augmentation_ratio == 0: augmentation mutates " \
            "train.data_x in place, which corrupts the full-series reconstruction."
        assert not self.args.prob_expert, \
            "calibrate_cqr_retrain requires quantile heads (--use_quantile_loss), which are " \
            "incompatible with --prob_expert."

        seq_len, pred_len = self.args.seq_len, self.args.pred_len
        window = getattr(self.args, 'retrain_window', 1000)
        interval = getattr(self.args, 'retrain_interval', None) or pred_len

        full_x, full_stamp, test_offset, _, test_data_obj = self._build_full_series()

        # One shared window for the model fit and the conformal quantile (ACI Sec. 2.2)
        calibrator = OnlineCQRQuantile(alpha=0.1, window_size=window)

        _, val_loader = self._get_data(flag='val')
        val_low, val_up, val_true = self._predict_quantiles(val_loader)
        calibrator.fit(val_low, val_up, val_true)

        n_test = len(test_data_obj)
        test_low = np.zeros((n_test,) + val_low.shape[1:], dtype=val_low.dtype)
        test_up = np.zeros_like(test_low)
        test_true = np.zeros_like(test_low)

        final_lowers, final_uppers, q_history = [], [], []
        last_q = None
        n_retrains = 0
        retrain_time = 0.0

        print(f"Starting Retrained CQR: window={window}, interval={interval}, "
              f"mode={getattr(self.args, 'retrain_mode', 'finetune')}, "
              f"epochs={getattr(self.args, 'retrain_epochs', 3)}, delay={pred_len} steps...")

        for t0 in range(0, n_test, interval):
            # Last sample whose target is fully observed at test row t0.
            frontier = test_offset + t0 - pred_len
            train_start = frontier - window + 1
            train_end = frontier + seq_len + pred_len  # exclusive

            if t0 > 0 and train_start >= 0 and (train_end - train_start) >= (seq_len + pred_len + 1):
                t_start = time.time()
                self._retrain_on_window(full_x, full_stamp, train_start, train_end)
                retrain_time += time.time() - t_start
                n_retrains += 1
                # Scores already in the FIFO stay valid: each was produced by the
                # model current at that time, out-of-sample w.r.t. its own target.

            t_hi = min(t0 + interval, n_test)
            chunk_start = test_offset + t0
            chunk_end = test_offset + (t_hi - 1) + seq_len + pred_len  # exclusive
            _, chunk_loader = self._window_loader(full_x, full_stamp, chunk_start, chunk_end)
            c_low, c_up, c_true = self._predict_quantiles(chunk_loader)

            test_low[t0:t_hi] = c_low[:t_hi - t0]
            test_up[t0:t_hi] = c_up[:t_hi - t0]
            test_true[t0:t_hi] = c_true[:t_hi - t0]

            for t in range(t0, t_hi):
                window_changed = ((t - 1 - pred_len) >= 0)

                if last_q is None or window_changed:
                    l, u, q = calibrator.predict_one_step(test_low[t], test_up[t])
                    last_q = q
                else:
                    q = last_q
                    l = test_low[t] - q
                    u = test_up[t] + q

                final_lowers.append(l)
                final_uppers.append(u)
                q_history.append(q)

                t_update = t - pred_len
                if t_update >= 0:
                    calibrator.update(test_low[t_update], test_up[t_update], test_true[t_update])

        final_lowers = np.array(final_lowers)
        final_uppers = np.array(final_uppers)

        if test_data_obj.scale and self.args.inverse:
            print("Applying Inverse Transform to Retrained CQR results...")
            shape = final_lowers.shape
            final_lowers = test_data_obj.inverse_transform(final_lowers.reshape(shape[0] * shape[1], -1)).reshape(shape)
            final_uppers = test_data_obj.inverse_transform(final_uppers.reshape(shape[0] * shape[1], -1)).reshape(shape)
            test_true = test_data_obj.inverse_transform(test_true.reshape(shape[0] * shape[1], -1)).reshape(shape)

        coverage = np.mean((test_true >= final_lowers) & (test_true <= final_uppers))
        width = np.mean(np.abs(final_uppers - final_lowers))

        print(f"\nRetrained CQR Results:")
        print(f"Retrains: {n_retrains}, total retrain time: {retrain_time:.1f}s")
        print(f"Mean q: {np.mean(q_history):.4f}")
        print(f"Coverage: {coverage:.4f}")
        print(f"Avg Width: {width:.4f}")

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open("result_calibration_cqr_retrain.txt", 'a') as f:
            f.write(f"{setting} (Retrained CQR alpha=0.1, mode={getattr(self.args, 'retrain_mode', 'finetune')}, "
                    f"window={window}, interval={interval}, epochs={getattr(self.args, 'retrain_epochs', 3)})\n")
            f.write(f"q_mean: {np.mean(q_history):.4f}, Coverage: {coverage:.4f}, Width: {width:.4f}, "
                    f"retrains: {n_retrains}, retrain_time: {retrain_time:.1f}s\n\n")

        return coverage, width

    # ------------------------------------------------------------------
    # ACI-adaptive variants of the fixed-alpha calibrators above.
    #
    # Each mirrors its base method with three changes: the calibrator comes from the
    # calibration/aci_*.py family, alpha/gamma come from --aci_alpha/--aci_gamma, and the
    # `last_q` cache is dropped. That last one is not an optimization choice -- the ACI
    # calibrators record the level they served inside interval_from_q so the pred_len-delayed
    # update can score against it, and the base loops build the interval inline on cache
    # hits, which would skip the record and desynchronize the queue permanently. The base
    # methods themselves are deliberately left untouched.
    # ------------------------------------------------------------------

    def calibrate_aci_cp(self, setting):
        print(">>>>>>> Start ACI CP Calibration (Sliding Window) >>>>>>>>>>>")

        path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
        self.model.eval()

        calibrator = ACICPCalibrator(alpha=self.args.aci_alpha, gamma=self.args.aci_gamma,
                                     window_size=1000)

        # Deliberate copy of calibrate_cp's get_deterministic_preds closure. It is nested in
        # that method and the base drivers are not modified by this change, so the two are
        # kept in sync by eye -- diff them if you touch either.
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

        print("Running Online ACI Calibration on Test set...")
        test_preds, test_trues, test_data_obj = get_deterministic_preds('test')

        final_lowers, final_uppers, q_history = [], [], []
        n_test = test_preds.shape[0]
        pred_len = self.args.pred_len

        for t in range(n_test):
            lower, upper, curr_q = calibrator.predict_one_step(test_preds[t])

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

        print(f"\nACI CP Results (gamma={calibrator.aci.gamma}):")
        print(f"Mean q (Absolute Error Quantile): {np.mean(q_history):.4f}")
        print(f"Coverage: {coverage:.4f}")
        print(f"Avg Width: {width:.4f}")
        print(calibrator.aci.summary_line())

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open("result_calibration_aci_cp_tsf.txt", 'a') as f:
            f.write(f"{setting} (ACI CP Sliding Window, alpha={calibrator.alpha}, "
                    f"gamma={calibrator.aci.gamma})\n")
            f.write(f"q_mean: {np.mean(q_history):.4f}, Coverage: {coverage:.4f}, "
                    f"Width: {width:.4f}{calibrator.aci.result_suffix()}\n\n")

        return coverage, width

    def calibrate_aci_cpvs(self, setting):
        print(">>>>>>> Start Calibration (ACI CP-VS) >>>>>>>>>>>")

        path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
        self.model.eval()

        calibrator = ACICPVSCalibrator(alpha=self.args.aci_alpha, gamma=self.args.aci_gamma,
                                       window_size=1000)

        # Deliberate copy of calibrate_cpvs's get_data_with_uncertainty closure. Note it is
        # NOT interchangeable with _collect_separated_uncertainty: that one aggregates via
        # _moe_aggregate and returns the two variance components separately, and its non-MoE
        # fallback differs (ale=1e-6, epi=0 -> sigma=1e-3, versus sigma=1e-6 here).
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

        final_lowers, final_uppers, q_history = [], [], []
        n_test = test_preds.shape[0]
        pred_len = self.args.pred_len

        for t in range(n_test):
            lower, upper, curr_q = calibrator.predict_one_step(test_preds[t], test_uncs[t])

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

        print(f"\nACI CP-VS Results (gamma={calibrator.aci.gamma}):")
        print(f"Mean q: {np.mean(q_history):.4f}")
        print(f"Coverage: {coverage:.4f}")
        print(f"Avg Width: {width:.4f}")
        print(calibrator.aci.summary_line())

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open("result_calibration_aci_cpvs_tsf.txt", 'a') as f:
            f.write(f"{setting} (ACI CP-VS Sliding Window, alpha={calibrator.alpha}, "
                    f"gamma={calibrator.aci.gamma})\n")
            f.write(f"q_mean: {np.mean(q_history):.4f}, Coverage: {coverage:.4f}, "
                    f"Width: {width:.4f}{calibrator.aci.result_suffix()}\n\n")

        return coverage, width

    def calibrate_aci_cqr(self, setting):
        print(">>>>>>> Start ACI CQR Calibration (Sliding Window) >>>>>>>>>>>")

        path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
        self.model.eval()

        calibrator = ACICQRCalibrator(alpha=self.args.aci_alpha, gamma=self.args.aci_gamma,
                                      window_size=1000)

        # _predict_quantiles is documented as mirroring calibrate_cqr's get_quantile_preds,
        # so it is reused rather than copied; only the dataset object has to be fetched too.
        def get_quantile_preds(flag):
            data_set, loader = self._get_data(flag=flag)
            low, up, true = self._predict_quantiles(loader)
            return low, up, true, data_set

        val_low, val_high, val_true, _ = get_quantile_preds('val')
        calibrator.fit(val_low, val_high, val_true)

        test_low, test_high, test_true, test_data_obj = get_quantile_preds('test')

        final_lowers, final_uppers, q_history = [], [], []
        pred_len = self.args.pred_len

        print(f"Starting ACI Sliding Window CQR with Delay of {pred_len} steps...")

        for t in range(len(test_true)):
            l, u, q = calibrator.predict_one_step(test_low[t], test_high[t])

            final_lowers.append(l)
            final_uppers.append(u)
            q_history.append(q)

            t_update = t - pred_len
            if t_update >= 0:
                calibrator.update(test_low[t_update], test_high[t_update], test_true[t_update])

        final_lowers = np.array(final_lowers)
        final_uppers = np.array(final_uppers)

        if test_data_obj.scale and self.args.inverse:
            print("Applying Inverse Transform to ACI CQR results...")
            shape = final_lowers.shape
            final_lowers = test_data_obj.inverse_transform(final_lowers.reshape(shape[0] * shape[1], -1)).reshape(shape)
            final_uppers = test_data_obj.inverse_transform(final_uppers.reshape(shape[0] * shape[1], -1)).reshape(shape)
            test_true = test_data_obj.inverse_transform(test_true.reshape(shape[0] * shape[1], -1)).reshape(shape)

        coverage = np.mean((test_true >= final_lowers) & (test_true <= final_uppers))
        width = np.mean(np.abs(final_uppers - final_lowers))

        print(f"\nACI CQR Quantile Results (gamma={calibrator.aci.gamma}):")
        print(f"Mean q: {np.mean(q_history):.4f}")
        print(f"Coverage: {coverage:.4f}")
        print(f"Avg Width: {width:.4f}")
        print(calibrator.aci.summary_line())

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open("result_calibration_aci_cqr_quantile_tsf.txt", 'a') as f:
            f.write(f"{setting} (ACI CQR Quantile, alpha={calibrator.alpha}, "
                    f"gamma={calibrator.aci.gamma}, Delayed Update)\n")
            f.write(f"q_mean: {np.mean(q_history):.4f}, Coverage: {coverage:.4f}, "
                    f"Width: {width:.4f}{calibrator.aci.result_suffix()}\n\n")

        return coverage, width

    def calibrate_aci_cqr_retrain(self, setting):
        """ACI on the rolling-retrain CQR.

        Gibbs & Candes Sec. 2.2 (refit the model on a trailing window) and Sec. 2.1 (adapt
        the target level online) are independent; the base method runs only the former, this
        runs both.
        """
        print(">>>>>>> Start ACI Retrained CQR Calibration (Rolling Window) >>>>>>>>>>>")

        path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
        self.model.eval()

        assert getattr(self.args, 'augmentation_ratio', 0) == 0, \
            "calibrate_aci_cqr_retrain requires augmentation_ratio == 0: augmentation mutates " \
            "train.data_x in place, which corrupts the full-series reconstruction."
        assert not self.args.prob_expert, \
            "calibrate_aci_cqr_retrain requires quantile heads (--use_quantile_loss), which are " \
            "incompatible with --prob_expert."

        seq_len, pred_len = self.args.seq_len, self.args.pred_len
        window = getattr(self.args, 'retrain_window', 1000)
        interval = getattr(self.args, 'retrain_interval', None) or pred_len

        full_x, full_stamp, test_offset, _, test_data_obj = self._build_full_series()

        calibrator = ACICQRCalibrator(alpha=self.args.aci_alpha, gamma=self.args.aci_gamma,
                                      window_size=window)

        _, val_loader = self._get_data(flag='val')
        val_low, val_up, val_true = self._predict_quantiles(val_loader)
        calibrator.fit(val_low, val_up, val_true)

        n_test = len(test_data_obj)
        test_low = np.zeros((n_test,) + val_low.shape[1:], dtype=val_low.dtype)
        test_up = np.zeros_like(test_low)
        test_true = np.zeros_like(test_low)

        final_lowers, final_uppers, q_history = [], [], []
        n_retrains = 0
        retrain_time = 0.0

        print(f"Starting ACI Retrained CQR: window={window}, interval={interval}, "
              f"mode={getattr(self.args, 'retrain_mode', 'finetune')}, "
              f"epochs={getattr(self.args, 'retrain_epochs', 3)}, "
              f"gamma={calibrator.aci.gamma}, delay={pred_len} steps...")

        for t0 in range(0, n_test, interval):
            frontier = test_offset + t0 - pred_len
            train_start = frontier - window + 1
            train_end = frontier + seq_len + pred_len  # exclusive

            if t0 > 0 and train_start >= 0 and (train_end - train_start) >= (seq_len + pred_len + 1):
                t_start = time.time()
                self._retrain_on_window(full_x, full_stamp, train_start, train_end)
                retrain_time += time.time() - t_start
                n_retrains += 1

            t_hi = min(t0 + interval, n_test)
            chunk_start = test_offset + t0
            chunk_end = test_offset + (t_hi - 1) + seq_len + pred_len  # exclusive
            _, chunk_loader = self._window_loader(full_x, full_stamp, chunk_start, chunk_end)
            c_low, c_up, c_true = self._predict_quantiles(chunk_loader)

            test_low[t0:t_hi] = c_low[:t_hi - t0]
            test_up[t0:t_hi] = c_up[:t_hi - t0]
            test_true[t0:t_hi] = c_true[:t_hi - t0]

            for t in range(t0, t_hi):
                l, u, q = calibrator.predict_one_step(test_low[t], test_up[t])

                final_lowers.append(l)
                final_uppers.append(u)
                q_history.append(q)

                t_update = t - pred_len
                if t_update >= 0:
                    calibrator.update(test_low[t_update], test_up[t_update], test_true[t_update])

        final_lowers = np.array(final_lowers)
        final_uppers = np.array(final_uppers)

        if test_data_obj.scale and self.args.inverse:
            print("Applying Inverse Transform to ACI Retrained CQR results...")
            shape = final_lowers.shape
            final_lowers = test_data_obj.inverse_transform(final_lowers.reshape(shape[0] * shape[1], -1)).reshape(shape)
            final_uppers = test_data_obj.inverse_transform(final_uppers.reshape(shape[0] * shape[1], -1)).reshape(shape)
            test_true = test_data_obj.inverse_transform(test_true.reshape(shape[0] * shape[1], -1)).reshape(shape)

        coverage = np.mean((test_true >= final_lowers) & (test_true <= final_uppers))
        width = np.mean(np.abs(final_uppers - final_lowers))

        print(f"\nACI Retrained CQR Results (gamma={calibrator.aci.gamma}):")
        print(f"Retrains: {n_retrains}, total retrain time: {retrain_time:.1f}s")
        print(f"Mean q: {np.mean(q_history):.4f}")
        print(f"Coverage: {coverage:.4f}")
        print(f"Avg Width: {width:.4f}")
        print(calibrator.aci.summary_line())

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open("result_calibration_aci_cqr_retrain_tsf.txt", 'a') as f:
            f.write(f"{setting} (ACI Retrained CQR, alpha={calibrator.alpha}, "
                    f"gamma={calibrator.aci.gamma}, mode={getattr(self.args, 'retrain_mode', 'finetune')}, "
                    f"window={window}, interval={interval}, epochs={getattr(self.args, 'retrain_epochs', 3)})\n")
            f.write(f"q_mean: {np.mean(q_history):.4f}, Coverage: {coverage:.4f}, Width: {width:.4f}, "
                    f"retrains: {n_retrains}, retrain_time: {retrain_time:.1f}s"
                    f"{calibrator.aci.result_suffix()}\n\n")

        return coverage, width

