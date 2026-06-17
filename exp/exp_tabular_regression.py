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
        all_preds = []
        all_trues = []
        all_weights = []
        all_stds = []

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(dataloader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)
                
                if isinstance(outputs, tuple):
                    pred = outputs[0]
                    weights = outputs[1]
                    std = outputs[2] if len(outputs) > 2 else torch.ones_like(pred)
                else:
                    pred = outputs
                    weights = torch.ones((pred.shape[0], 1)).to(self.device)
                    std = torch.ones_like(pred)

                all_preds.append(pred.detach().cpu())
                all_trues.append(batch_y.detach().cpu())
                all_weights.append(weights.detach().cpu())
                all_stds.append(std.detach().cpu())

        return torch.cat(all_preds), torch.cat(all_trues), torch.cat(all_weights), torch.cat(all_stds)

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)
                
                if isinstance(outputs, tuple):
                    pred = outputs[0]
                    std = outputs[2] if len(outputs) > 2 else None
                else:
                    pred = outputs
                    std = None
                
                if hasattr(self.args, 'prob_expert') and self.args.prob_expert and std is not None:
                    loss = criterion(pred, batch_y, std ** 2)
                else:
                    loss = criterion(pred, batch_y)
                    
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
                
                if isinstance(outputs, tuple):
                    pred = outputs[0]
                    std = outputs[2] if len(outputs) > 2 else None
                else:
                    pred = outputs
                    std = None
                
                if hasattr(self.args, 'prob_expert') and self.args.prob_expert and std is not None:
                    loss = criterion(pred, batch_y, std ** 2)
                else:
                    loss = criterion(pred, batch_y)
                    
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

        test_preds, test_trues, _, _ = self._collect_predictions(test_loader)

        test_preds = test_preds.numpy()
        test_trues = test_trues.numpy()

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

        cal_preds, cal_trues, cal_weights, _ = self._collect_predictions(cal_loader)
        test_preds, test_trues, test_weights, _ = self._collect_predictions(test_loader)

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

        np.save(folder_path + 'intervals.npy', intervals)
        return coverage, width

    def calibrate_cpvs(self, setting):
        path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
        self.model.eval()

        cal_data, cal_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        cal_preds, cal_trues, _, cal_stds = self._collect_predictions(cal_loader)
        test_preds, test_trues, _, test_stds = self._collect_predictions(test_loader)

        calibrator = CP_VS_Static_Calibrator(alpha=0.1)
        calibrator.fit(cal_preds, cal_trues, cal_stds)
        intervals = calibrator.predict(test_preds, test_stds)

        intervals = intervals.numpy()
        test_trues = test_trues.numpy()

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

        np.save(folder_path + 'intervals.npy', intervals)
        return coverage, width