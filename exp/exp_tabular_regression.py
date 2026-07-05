import os
import time
import torch
import torch.nn as nn
from torch import optim
import numpy as np
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from calibration.tabular_regression.MoECP_Calibrator import MoECP_Calibrator
from calibration.tabular_regression.CPVS_regression import CP_VS_Static_Calibrator
from calibration.tabular_regression.Aleatoric_CPVS_Calibrator import AleatoricCPVSCalibrator
from calibration.tabular_regression.Aleatoric_Scale_Calibrator import AleatoricScaleCalibrator
from calibration.tabular_regression.Standard_CP_Calibrator import Standard_CP_Calibrator

class Exp_Tabular_Regression(Exp_Basic):
    def __init__(self, args):
        super(Exp_Tabular_Regression, self).__init__(args)
        self.max_grad_norm = 1

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        from data_provider.data_factory import data_provider
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if hasattr(self.args, 'prob_expert') and self.args.prob_expert:
            criterion = nn.GaussianNLLLoss()
        else:
            criterion = nn.MSELoss()
        return criterion
    def _collect_predictions(self, dataloader):
        self.model.eval()
        all_preds, all_trues, all_weights = [], [], []
        all_stds_total, all_stds_aleat, all_stds_epist = [], [], []

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(dataloader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)
                
                if isinstance(outputs, tuple):
                    pred = outputs[0]
                    weights = outputs[1]
                    std_total = outputs[2] if len(outputs) > 2 else torch.ones_like(pred)
                    std_aleat = outputs[3] if len(outputs) > 3 else std_total
                    std_epist = outputs[4] if len(outputs) > 4 else torch.zeros_like(pred)
                else:
                    pred = outputs
                    weights = torch.ones((pred.shape[0], 1)).to(self.device)
                    std_total = torch.ones_like(pred)
                    std_aleat = torch.ones_like(pred)
                    std_epist = torch.zeros_like(pred)

                all_preds.append(pred.detach().cpu())
                all_trues.append(batch_y.detach().cpu())
                all_weights.append(weights.detach().cpu())
                all_stds_total.append(std_total.detach().cpu())
                all_stds_aleat.append(std_aleat.detach().cpu())
                all_stds_epist.append(std_epist.detach().cpu())

        # Returns 6 tensors now
        return (torch.cat(all_preds), torch.cat(all_trues), torch.cat(all_weights), 
                torch.cat(all_stds_total), torch.cat(all_stds_aleat), torch.cat(all_stds_epist))

    def _select_criterion(self):
        # שינוי חשוב: reduction='none' מאפשר חישוב נפרד לכל מומחה
        if hasattr(self.args, 'prob_expert') and self.args.prob_expert:
            criterion = nn.GaussianNLLLoss(reduction='none') 
        else:
            criterion = nn.MSELoss(reduction='none')
        return criterion

    def moe_loss(self, expert_outputs, expert_unc, gating_weights, batch_y, criterion):
        if hasattr(self.args, 'prob_expert') and self.args.prob_expert:
            eps = 1e-8
            # Mixture of Gaussians (MoG) NLL
            dist = torch.distributions.Normal(loc=expert_outputs, scale=torch.sqrt(expert_unc + eps))
            log_prob = dist.log_prob(batch_y.unsqueeze(1)) 
            log_weights = torch.log(gating_weights + eps).unsqueeze(-1) 
            
            # חיבור הסתברויות לוגריתמי מונע קריסה מעודד גיוון מומחים
            log_likelihood = torch.logsumexp(log_weights + log_prob, dim=1)
            loss = -torch.mean(log_likelihood)
        else:
            # Weighted MSE Loss
            expert_loss = criterion(expert_outputs, batch_y.unsqueeze(1)) 
            loss = torch.sum(gating_weights.unsqueeze(-1) * expert_loss, dim=1).mean()
            
            # הוספת Load Balancing למומחים לא-הסתברותיים כדי למנוע קריסה למומחה יחיד
            if not self.args.unc_gating:
                avg_weight = gating_weights.mean(dim=0)
                load_balance_loss = self.args.num_experts * torch.sum(avg_weight * avg_weight)
                loss += 0.05 * load_balance_loss
        return loss

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)
                
                if isinstance(outputs, tuple) and len(outputs) >= 7:
                    # הפירוק החדש שכולל את תפוקות המומחים
                    pred, gating_weights, total_std, aleat_std, epist_std, expert_outputs, expert_unc = outputs
                    loss = self.moe_loss(expert_outputs, expert_unc, gating_weights, batch_y, criterion)
                elif isinstance(outputs, tuple):
                    pred = outputs[0]
                    std = outputs[2] if len(outputs) > 2 else None
                    if hasattr(self.args, 'prob_expert') and self.args.prob_expert and std is not None:
                        loss = criterion(pred, batch_y, std ** 2).mean()
                    else:
                        loss = criterion(pred, batch_y).mean()
                else:
                    pred = outputs
                    loss = criterion(pred, batch_y).mean()
                    
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
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            train_loss = []
            self.model.train()
            
            for i, (batch_x, batch_y) in enumerate(train_loader):
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)
                
                if isinstance(outputs, tuple) and len(outputs) >= 7:
                    pred, gating_weights, total_std, aleat_std, epist_std, expert_outputs, expert_unc = outputs
                    loss = self.moe_loss(expert_outputs, expert_unc, gating_weights, batch_y, criterion)
                elif isinstance(outputs, tuple):
                    pred = outputs[0]
                    std = outputs[2] if len(outputs) > 2 else None
                    if hasattr(self.args, 'prob_expert') and self.args.prob_expert and std is not None:
                        loss = criterion(pred, batch_y, std ** 2).mean()
                    else:
                        loss = criterion(pred, batch_y).mean()
                else:
                    pred = outputs
                    loss = criterion(pred, batch_y).mean()
                    
                train_loss.append(loss.item())
                loss.backward()
                
                if hasattr(self.args, 'prob_expert') and self.args.prob_expert and self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                model_optim.step()

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")

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

        # איסוף כל הנתונים מהמודל (6 משתנים)
        test_preds, test_trues, test_weights, test_stds_total, test_stds_aleat, test_stds_epist = self._collect_predictions(test_loader)

        test_preds = test_preds.numpy()
        test_trues = test_trues.numpy()
        test_weights = test_weights.numpy()
        
        # --- תוספת: ניתוח המומחים וחוסר הוודאות ---
        if hasattr(self.args, 'prob_expert') and self.args.prob_expert:
            var_aleat = test_stds_aleat.numpy() ** 2
            var_epist = test_stds_epist.numpy() ** 2
            
            # 1. חישוב יחס השונות (Epistemic / Aleatoric)
            # כמה מהשגיאה נובעת מחוסר הסכמה בין המומחים לעומת רעש בדאטה
            ratio = var_epist / (var_aleat + 1e-8)
            epistemic_contribution = var_epist / (var_aleat + var_epist + 1e-8)
            
            print("\n" + "="*40)
            print("--- Uncertainty & Variance Analysis ---")
            print(f"Avg Aleatoric Variance: {np.mean(var_aleat):.4f}")
            print(f"Avg Epistemic Variance: {np.mean(var_epist):.4f}")
            print(f"Avg Epistemic/Aleatoric Ratio: {np.mean(ratio):.4f}")
            print(f"Avg Epistemic Contribution to Total Var: {np.mean(epistemic_contribution)*100:.2f}%")
            
        print("\n--- MoE Gating Analysis ---")
        # 2. ניתוח דינמיות המשקלים (האם הנתב באמת מחלק עבודה?)
        avg_weights = np.mean(test_weights, axis=0)
        std_weights = np.std(test_weights, axis=0)
        
        # מי המומחה שניצח (קיבל את המשקל הגבוה ביותר) בכל דגימה?
        winning_expert = np.argmax(test_weights, axis=1)
        unique, counts = np.unique(winning_expert, return_counts=True)
        expert_counts = dict(zip(unique, counts))
        
        print(f"Average Gating Weights per expert: {avg_weights}")
        print(f"Std of Gating Weights per expert: {std_weights}")
        print(f"Winning expert distribution (Count of samples each expert 'won'):")
        for exp_idx, count in expert_counts.items():
            print(f"  Expert {exp_idx}: {count} samples ({(count/len(test_preds))*100:.1f}%)")
        print("="*40 + "\n")
        # ----------------------------------------

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
        test_trues = test_trues.numpy()

        coverage = np.mean((test_trues >= intervals[:, 0]) & (test_trues <= intervals[:, 1]))
        width = np.mean(intervals[:, 1] - intervals[:, 0])
        
        print(f"\nMoECP Results:")
        print(f"Coverage: {coverage:.4f}")
        print(f"Avg Width: {width:.4f}")

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open("result_calibration_moecp.txt", 'a') as f:
            f.write(f"{setting} (MoECP)\n")
            f.write(f"Coverage: {coverage:.4f}, Width: {width:.4f}\n\n")

        np.save(folder_path + 'intervals_moecp.npy', intervals)
        return coverage, width

    def calibrate_cpvs(self, setting):
        path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
        self.model.eval()

        cal_data, cal_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # חילוץ 6 הערכים: אנחנו שומרים את השונות הכוללת במשתנה ייעודי
        cal_preds, cal_trues, _, cal_stds_total, _, _ = self._collect_predictions(cal_loader)
        test_preds, test_trues, _, test_stds_total, _, _ = self._collect_predictions(test_loader)

        # שימוש בשונות הכוללת (Total Std) עבור כיול CP_VS קלאסי
        calibrator = CP_VS_Static_Calibrator(alpha=0.1)
        calibrator.fit(cal_preds, cal_trues, cal_stds_total)
        intervals = calibrator.predict(test_preds, test_stds_total)

        intervals = intervals.numpy()
        test_trues = test_trues.squeeze().numpy()

        coverage = np.mean((test_trues >= intervals[:, 0]) & (test_trues <= intervals[:, 1]))
        width = np.mean(intervals[:, 1] - intervals[:, 0])
        
        print(f"\nCP_VS Static Results:")
        print(f"Coverage: {coverage:.4f}")
        print(f"Avg Width: {width:.4f}")

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open("result_calibration_cpvs.txt", 'a') as f:
            f.write(f"{setting} (CP_VS Static)\n")
            f.write(f"Coverage: {coverage:.4f}, Width: {width:.4f}\n\n")

        np.save(folder_path + 'intervals_cpvs.npy', intervals)
        return coverage, width
    
    def calibrate_cpvs_aleatoric(self, setting):
        path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
        self.model.eval()

        cal_data, cal_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # Extract predictions and isolate aleatoric uncertainty
        cal_preds, cal_trues, _, _, cal_stds_aleat, _ = self._collect_predictions(cal_loader)
        test_preds, test_trues, _, _, test_stds_aleat, _ = self._collect_predictions(test_loader)

        # Deploy the dedicated Aleatoric CP-VS calibrator
        calibrator = AleatoricCPVSCalibrator(alpha=0.1)
        calibrator.fit(cal_preds, cal_trues, cal_stds_aleat)
        intervals = calibrator.predict(test_preds, test_stds_aleat)

        intervals = intervals.numpy()
        test_trues = test_trues.squeeze().numpy()

        coverage = np.mean((test_trues >= intervals[:, 0]) & (test_trues <= intervals[:, 1]))
        width = np.mean(intervals[:, 1] - intervals[:, 0])
        
        print(f"\nCP_VS Aleatoric Results:")
        print(f"Coverage: {coverage:.4f}")
        print(f"Avg Width: {width:.4f}")

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open("result_calibration_cpvs_aleatoric.txt", 'a') as f:
            f.write(f"{setting} (CP_VS Aleatoric)\n")
            f.write(f"Coverage: {coverage:.4f}, Width: {width:.4f}\n\n")

        np.save(folder_path + 'intervals_cpvs_aleatoric.npy', intervals)
        return coverage, width

    def calibrate_cp_aleatoric_scale(self, setting):
        path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
        self.model.eval()

        cal_data, cal_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # Collect complete predictions and uncertainty profiles
        cal_preds, cal_trues, _, _, cal_stds_aleat, cal_stds_epist = self._collect_predictions(cal_loader)
        test_preds, test_trues, _, _, test_stds_aleat, test_stds_epist = self._collect_predictions(test_loader)

        # Deploy the new standalone AleatoricScaleCalibrator
        calibrator = AleatoricScaleCalibrator(alpha=0.1)
        calibrator.fit(cal_preds, cal_trues, cal_stds_aleat, cal_stds_epist)
        intervals = calibrator.predict(test_preds, test_stds_aleat, test_stds_epist)

        intervals = intervals.numpy()
        test_trues = test_trues.squeeze().numpy()

        coverage = np.mean((test_trues >= intervals[:, 0]) & (test_trues <= intervals[:, 1]))
        width = np.mean(intervals[:, 1] - intervals[:, 0])
        
        print(f"\nCP Aleatoric Scale Results (Learned q^2 = {calibrator.q_sq:.4f}):")
        print(f"Coverage: {coverage:.4f}")
        print(f"Avg Width: {width:.4f}")

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open("result_calibration_cp_aleatoric_scale.txt", 'a') as f:
            f.write(f"{setting} (CP Aleatoric Scale)\n")
            f.write(f"Coverage: {coverage:.4f}, Width: {width:.4f}, q_sq: {calibrator.q_sq:.4f}\n\n")

        np.save(folder_path + 'intervals_cp_aleatoric_scale.npy', intervals)
        return coverage, width
    
    def calibrate_standard_cp(self, setting):
        path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
        self.model.eval()

        cal_data, cal_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # חילוץ נתונים (שים לב שאנחנו מתעלמים מכל מדדי השונות עם _, כי CP רגיל לא צריך אותם)
        cal_preds, cal_trues, _, _, _, _ = self._collect_predictions(cal_loader)
        test_preds, test_trues, _, _, _, _ = self._collect_predictions(test_loader)

        # הפעלת הכיול הסטנדרטי
        calibrator = Standard_CP_Calibrator(alpha=0.1)
        calibrator.fit(cal_preds, cal_trues)
        intervals = calibrator.predict(test_preds)

        intervals = intervals if isinstance(intervals, np.ndarray) else intervals.numpy()
        test_trues = test_trues.squeeze().numpy() if torch.is_tensor(test_trues) else test_trues.squeeze()

        coverage = np.mean((test_trues >= intervals[:, 0]) & (test_trues <= intervals[:, 1]))
        width = np.mean(intervals[:, 1] - intervals[:, 0])
        
        print(f"\nStandard CP Results:")
        print(f"Coverage: {coverage:.4f}")
        print(f"Avg Width: {width:.4f}")

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open("result_calibration_standard_cp.txt", 'a') as f:
            f.write(f"{setting} (Standard CP)\n")
            f.write(f"Coverage: {coverage:.4f}, Width: {width:.4f}\n\n")

        np.save(folder_path + 'intervals_standard_cp.npy', intervals)
        return coverage, width