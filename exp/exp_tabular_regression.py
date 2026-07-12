import os
import time
import math
import random
import torch
import torch.nn as nn
from torch import optim
import numpy as np

from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping
from models import SimpleMLP, RegressionMoE

from calibration.tabular_regression.MoECP_Calibrator import MoECP_Calibrator
from calibration.tabular_regression.CPVS_regression import CP_VS_Static_Calibrator
from calibration.tabular_regression.Aleatoric_CPVS_Calibrator import AleatoricCPVSCalibrator
from calibration.tabular_regression.Aleatoric_Scale_Calibrator import AleatoricScaleCalibrator
from calibration.tabular_regression.Standard_CP_Calibrator import Standard_CP_Calibrator

def set_seed(seed=45):
    """קיבוע אקראיות מוחלט כדי להבטיח הוגנות באתחול בין המודלים, בדיוק כמו בסינתטי"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class Exp_Tabular_Regression(Exp_Basic):
    def __init__(self, args):
        super(Exp_Tabular_Regression, self).__init__(args)

    def _build_model(self):
        # שימוש במחלקות המודולריות בדיוק כמו ב-run_synthetic.py
        if self.args.num_experts > 1:
            model = RegressionMoE.Model(self.args, SimpleMLP.Model).float()
        else:
            model = SimpleMLP.Model(self.args).float()
            
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            
        return model.to(self.device)

    def _get_data(self, flag):
        from data_provider.data_factory import data_provider
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _calculate_loss(self, batch_x, batch_y):
        """
        לוגיקת חישוב ה-Loss מועתקת מ-run_synthetic.py תוך התאמה בטוחה למימדים.
        """
        # הבטחה ש-batch_y הוא לפחות [batch_size, 1] כדי שיתאים לפלט המומחים
        if batch_y.dim() == 1:
            batch_y = batch_y.view(-1, 1)
            
        if self.args.num_experts > 1:
            expert_out, expert_unc, weights = self.model(batch_x)
            
            # Load-balancing penalty (Shazeer et al. 2017 importance loss): minimized when the
            # batch-average gating weight per expert is uniform (1/K). Not part of the MoE-CP
            # paper's stated loss (Appendix A.4.1 only mentions MSE), but without it the softmax
            # gate collapses onto a single expert on these datasets, which starves the MoE of
            # capacity and destroys MoE-CP's similarity-weighting mechanism (near-one-hot gating
            # vectors give ~0 weight to almost all calibration points). Deliberate deviation from
            # the paper's literal description, applied identically to all three loss branches.
            avg_weight = weights.squeeze(-1).mean(dim=0)
            load_balance_loss = 0.05 * self.args.num_experts * torch.sum(avg_weight * avg_weight)

            if self.args.prob_expert and not getattr(self.args, 'unc_gating', False):
                # 1. Classic MoG Loss
                log_probs = []
                for i in range(self.args.num_experts):
                    mean = expert_out[:, i, :]
                    var = torch.clamp(expert_unc[:, i, :], min=0.01)
                    w = weights[:, i, :]

                    log_w = torch.log(w + 1e-8)
                    log_norm = -0.5 * math.log(2 * math.pi) - 0.5 * torch.log(var) - 0.5 * ((batch_y - mean)**2) / var
                    log_probs.append(log_w + log_norm)

                log_probs_tensor = torch.stack(log_probs, dim=1)
                loss = -torch.logsumexp(log_probs_tensor, dim=1).mean() + load_balance_loss

            elif self.args.prob_expert and getattr(self.args, 'unc_gating', False):
                # 2. MoGU Loss
                weighted_loss = 0.0
                for i in range(self.args.num_experts):
                    mean = expert_out[:, i, :]
                    var = torch.clamp(expert_unc[:, i, :], min=0.01)
                    w = weights[:, i, :]

                    expert_loss = 0.5 * (torch.log(var) + ((batch_y - mean)**2) / var)
                    weighted_loss += w * expert_loss
                loss = weighted_loss.mean() + load_balance_loss

            else:
                # 3. Standard MoE Loss
                weighted_loss = 0.0
                for i in range(self.args.num_experts):
                    mean = expert_out[:, i, :]
                    w = weights[:, i, :]

                    expert_loss = (batch_y - mean)**2
                    weighted_loss += w * expert_loss
                loss = weighted_loss.mean() + load_balance_loss
        else:
            # Single Expert
            if self.args.prob_expert:
                mean, var = self.model(batch_x)
                var = var + 1e-8
                loss = 0.5 * (torch.log(var) + ((batch_y - mean)**2) / var).mean()
            else:
                mean = self.model(batch_x)
                loss = nn.MSELoss()(mean, batch_y)
                
        return loss

    def _collect_predictions(self, dataloader):
        """
        פונקציה זו אוספת את פלטי המודל הטהור ומחשבת דינמית את מדדי אי-הוודאות 
        הנדרשים על ידי הקליברטורים.
        """
        self.model.eval()
        all_preds, all_trues, all_weights = [], [], []
        all_stds_total, all_stds_aleat, all_stds_epist = [], [], []

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(dataloader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if self.args.num_experts > 1:
                    expert_out, expert_unc, weights = self.model(batch_x)
                    
                    # חישוב התחזית המצטברת בדיוק כמו ב-synthetic
                    pred = torch.sum(expert_out * weights, dim=1)
                    
                    if self.args.prob_expert:
                        # פירוק שונות (Law of Total Variance)
                        aleat_var = torch.sum(weights * expert_unc, dim=1)
                        epist_var = torch.sum(weights * (expert_out - pred.unsqueeze(1))**2, dim=1)
                        
                        std_total = torch.sqrt(aleat_var + epist_var)
                        std_aleat = torch.sqrt(aleat_var)
                        std_epist = torch.sqrt(epist_var)
                    else:
                        std_total = torch.std(expert_out, dim=1)
                        std_aleat = std_total
                        std_epist = torch.zeros_like(std_total)
                        
                    weights = weights.squeeze(-1)
                else:
                    # מומחה יחיד
                    if self.args.prob_expert:
                        pred, var = self.model(batch_x)
                        std_total = torch.sqrt(var + 1e-8)
                        std_aleat = std_total
                        std_epist = torch.zeros_like(std_total)
                    else:
                        pred = self.model(batch_x)
                        std_total = torch.ones_like(pred)
                        std_aleat = std_total
                        std_epist = torch.zeros_like(std_total)
                        
                    weights = torch.ones((pred.shape[0], 1)).to(self.device)

                all_preds.append(pred.squeeze(-1).detach().cpu())
                all_trues.append(batch_y.detach().cpu())
                all_weights.append(weights.detach().cpu())
                all_stds_total.append(std_total.squeeze(-1).detach().cpu())
                all_stds_aleat.append(std_aleat.squeeze(-1).detach().cpu())
                all_stds_epist.append(std_epist.squeeze(-1).detach().cpu())

        return (torch.cat(all_preds), torch.cat(all_trues), torch.cat(all_weights), 
                torch.cat(all_stds_total), torch.cat(all_stds_aleat), torch.cat(all_stds_epist))

    def vali(self, vali_data, vali_loader):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                loss = self._calculate_loss(batch_x, batch_y)
                total_loss.append(loss.item())
                
        return np.average(total_loss)

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()

        set_seed(45)

        for epoch in range(self.args.train_epochs):
            train_loss = []
            self.model.train()
            
            for i, (batch_x, batch_y) in enumerate(train_loader):
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                loss = self._calculate_loss(batch_x, batch_y)

                train_loss.append(loss.item())
                loss.backward()
                if getattr(self.args, 'max_grad_norm', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                model_optim.step()

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader)
            test_loss = self.vali(test_data, test_loader)

            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        test_preds, test_trues, test_weights, test_stds_total, test_stds_aleat, test_stds_epist = self._collect_predictions(test_loader)

        test_preds = test_preds.numpy()
        test_trues = test_trues.squeeze().numpy()
        test_weights = test_weights.numpy()
        
        if hasattr(self.args, 'prob_expert') and self.args.prob_expert and self.args.num_experts > 1:
            var_aleat = test_stds_aleat.numpy() ** 2
            var_epist = test_stds_epist.numpy() ** 2
            
            eps_val = 1e-8
            ratio = var_epist / (var_aleat + eps_val)
            epistemic_contribution = var_epist / (var_aleat + var_epist + eps_val)
            
            print("\n" + "="*40)
            print("--- Uncertainty & Variance Analysis ---")
            print(f"Avg Aleatoric Variance: {np.mean(var_aleat):.4f}")
            print(f"Avg Epistemic Variance: {np.mean(var_epist):.4f}")
            print(f"Avg Epistemic/Aleatoric Ratio: {np.mean(ratio):.4f}")
            print(f"Avg Epistemic Contribution to Total Var: {np.mean(epistemic_contribution)*100:.2f}%")
            
        print("\n--- MoE Gating Analysis ---")
        if self.args.num_experts > 1:
            avg_weights = np.mean(test_weights, axis=0)
            std_weights = np.std(test_weights, axis=0)
            winning_expert = np.argmax(test_weights, axis=1)
            unique, counts = np.unique(winning_expert, return_counts=True)
            expert_counts = dict(zip(unique, counts))
            
            print(f"Average Gating Weights per expert: {avg_weights}")
            print(f"Std of Gating Weights per expert: {std_weights}")
            print(f"Winning expert distribution (Count of samples each expert 'won'):")
            for exp_idx, count in expert_counts.items():
                print(f"  Expert {exp_idx}: {count} samples ({(count/len(test_preds))*100:.1f}%)")
        else:
            print("Single expert model. Gating analysis not applicable.")
        print("="*40 + "\n")

        mse = np.mean((test_preds - test_trues) ** 2)
        mae = np.mean(np.abs(test_preds - test_trues))
        print(f'Test Results - MSE: {mse:.4f}, MAE: {mae:.4f}')

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'true.npy', test_trues)
        np.save(folder_path + 'pred.npy', test_preds)
        return

    def calibrate_moecp(self, setting):
        path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
        self.model.eval()

        cal_data, cal_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        cal_preds, cal_trues, cal_weights, _, _, _ = self._collect_predictions(cal_loader)
        test_preds, test_trues, test_weights, _, _, _ = self._collect_predictions(test_loader)
        
        calibrator = MoECP_Calibrator(alpha=0.1, temperature=self.args.tau)
        calibrator.fit(cal_preds, cal_trues, cal_weights)
        intervals = calibrator.predict(test_preds, test_weights)

        intervals = intervals.numpy()
        test_trues = test_trues.squeeze().numpy()

        coverage = np.mean((test_trues >= intervals[:, 0]) & (test_trues <= intervals[:, 1]))
        width = np.mean(intervals[:, 1] - intervals[:, 0])
        median_width = np.median(intervals[:, 1] - intervals[:, 0])

        print(f"\nMoECP Results:")
        print(f"Coverage: {coverage:.4f}")
        print(f"Avg Width: {width:.4f}")
        print(f"Median Width: {median_width:.4f}")

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open("result_calibration_moecp.txt", 'a') as f:
            f.write(f"{setting} (MoECP)\n")
            f.write(f"Coverage: {coverage:.4f}, Width: {width:.4f}, Median Width: {median_width:.4f}\n\n")

        np.save(folder_path + 'intervals_moecp.npy', intervals)
        return coverage, width

    def calibrate_cpvs(self, setting):
        path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
        self.model.eval()

        cal_data, cal_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        cal_preds, cal_trues, _, cal_stds_total, _, _ = self._collect_predictions(cal_loader)
        test_preds, test_trues, _, test_stds_total, _, _ = self._collect_predictions(test_loader)

        calibrator = CP_VS_Static_Calibrator(alpha=0.1)
        calibrator.fit(cal_preds, cal_trues, cal_stds_total)
        intervals = calibrator.predict(test_preds, test_stds_total)

        intervals = intervals.numpy()
        test_trues = test_trues.squeeze().numpy()

        coverage = np.mean((test_trues >= intervals[:, 0]) & (test_trues <= intervals[:, 1]))
        width = np.mean(intervals[:, 1] - intervals[:, 0])
        median_width = np.median(intervals[:, 1] - intervals[:, 0])

        print(f"\nCP_VS Static Results:")
        print(f"Coverage: {coverage:.4f}")
        print(f"Avg Width: {width:.4f}")
        print(f"Median Width: {median_width:.4f}")

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open("result_calibration_cpvs.txt", 'a') as f:
            f.write(f"{setting} (CP_VS Static)\n")
            f.write(f"Coverage: {coverage:.4f}, Width: {width:.4f}, Median Width: {median_width:.4f}\n\n")

        np.save(folder_path + 'intervals_cpvs.npy', intervals)
        return coverage, width
    
    def calibrate_cpvs_aleatoric(self, setting):
        path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
        self.model.eval()

        cal_data, cal_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        cal_preds, cal_trues, _, _, cal_stds_aleat, _ = self._collect_predictions(cal_loader)
        test_preds, test_trues, _, _, test_stds_aleat, _ = self._collect_predictions(test_loader)

        calibrator = AleatoricCPVSCalibrator(alpha=0.1)
        calibrator.fit(cal_preds, cal_trues, cal_stds_aleat)
        intervals = calibrator.predict(test_preds, test_stds_aleat)

        intervals = intervals.numpy()
        test_trues = test_trues.squeeze().numpy()

        coverage = np.mean((test_trues >= intervals[:, 0]) & (test_trues <= intervals[:, 1]))
        width = np.mean(intervals[:, 1] - intervals[:, 0])
        median_width = np.median(intervals[:, 1] - intervals[:, 0])

        print(f"\nCP_VS Aleatoric Results:")
        print(f"Coverage: {coverage:.4f}")
        print(f"Avg Width: {width:.4f}")
        print(f"Median Width: {median_width:.4f}")

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open("result_calibration_cpvs_aleatoric.txt", 'a') as f:
            f.write(f"{setting} (CP_VS Aleatoric)\n")
            f.write(f"Coverage: {coverage:.4f}, Width: {width:.4f}, Median Width: {median_width:.4f}\n\n")

        np.save(folder_path + 'intervals_cpvs_aleatoric.npy', intervals)
        return coverage, width

    def calibrate_cp_aleatoric_scale(self, setting):
        path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
        self.model.eval()

        cal_data, cal_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        cal_preds, cal_trues, _, _, cal_stds_aleat, cal_stds_epist = self._collect_predictions(cal_loader)
        test_preds, test_trues, _, _, test_stds_aleat, test_stds_epist = self._collect_predictions(test_loader)

        calibrator = AleatoricScaleCalibrator(alpha=0.1)
        calibrator.fit(cal_preds, cal_trues, cal_stds_aleat, cal_stds_epist)
        intervals = calibrator.predict(test_preds, test_stds_aleat, test_stds_epist)

        intervals = intervals.numpy()
        test_trues = test_trues.squeeze().numpy()

        coverage = np.mean((test_trues >= intervals[:, 0]) & (test_trues <= intervals[:, 1]))
        width = np.mean(intervals[:, 1] - intervals[:, 0])
        median_width = np.median(intervals[:, 1] - intervals[:, 0])

        print(f"\nCP Aleatoric Scale Results (Learned q^2 = {calibrator.q_sq:.4f}):")
        print(f"Coverage: {coverage:.4f}")
        print(f"Avg Width: {width:.4f}")
        print(f"Median Width: {median_width:.4f}")

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open("result_calibration_cp_aleatoric_scale.txt", 'a') as f:
            f.write(f"{setting} (CP Aleatoric Scale)\n")
            f.write(f"Coverage: {coverage:.4f}, Width: {width:.4f}, Median Width: {median_width:.4f}, q_sq: {calibrator.q_sq:.4f}\n\n")

        np.save(folder_path + 'intervals_cp_aleatoric_scale.npy', intervals)
        return coverage, width
    
    def calibrate_standard_cp(self, setting):
        path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
        self.model.eval()

        cal_data, cal_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        cal_preds, cal_trues, _, _, _, _ = self._collect_predictions(cal_loader)
        test_preds, test_trues, _, _, _, _ = self._collect_predictions(test_loader)

        calibrator = Standard_CP_Calibrator(alpha=0.1)
        calibrator.fit(cal_preds, cal_trues)
        intervals = calibrator.predict(test_preds)

        intervals = intervals if isinstance(intervals, np.ndarray) else intervals.numpy()
        test_trues = test_trues.squeeze().numpy() if torch.is_tensor(test_trues) else test_trues.squeeze()

        coverage = np.mean((test_trues >= intervals[:, 0]) & (test_trues <= intervals[:, 1]))
        width = np.mean(intervals[:, 1] - intervals[:, 0])
        median_width = np.median(intervals[:, 1] - intervals[:, 0])

        print(f"\nStandard CP Results:")
        print(f"Coverage: {coverage:.4f}")
        print(f"Avg Width: {width:.4f}")
        print(f"Median Width: {median_width:.4f}")

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open("result_calibration_standard_cp.txt", 'a') as f:
            f.write(f"{setting} (Standard CP)\n")
            f.write(f"Coverage: {coverage:.4f}, Width: {width:.4f}, Median Width: {median_width:.4f}\n\n")

        np.save(folder_path + 'intervals_standard_cp.npy', intervals)
        return coverage, width