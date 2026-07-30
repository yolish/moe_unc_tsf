import os
import time
import math
import random
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import scipy.stats

from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping
from utils.losses import QuantileLoss
from models import SimpleMLP, RegressionMoE

from calibration.tabular_regression.MoECP_Calibrator import MoECP_Calibrator
from calibration.tabular_regression.MoCE_Calibrator import MoCE_Calibrator
from calibration.tabular_regression.CPVS_regression import CP_VS_Static_Calibrator
from calibration.tabular_regression.Aleatoric_CPVS_Calibrator import AleatoricCPVSCalibrator
from calibration.tabular_regression.Aleatoric_Scale_Calibrator import AleatoricScaleCalibrator
from calibration.tabular_regression.Standard_CP_Calibrator import Standard_CP_Calibrator
from calibration.tabular_regression.CQR_Calibrator import CQR_Calibrator
from calibration.tabular_regression.Adaptive_Variance_Calibrator import AdaptiveVarianceCalibrator
from calibration.tabular_regression.MoG_HPD_Calibrator import MoG_HPD_Calibrator
from calibration.tabular_regression.STA_HPD_Calibrator import STAHPDCalibrator
from calibration.tabular_regression.SETA_HPD_Calibrator import SETAHPDCalibrator
from calibration.tabular_regression.MELD_HPD_Calibrator import MELDHPDCalibrator
from calibration.tabular_regression.EASE_HPD_Calibrator import EASEHPDCalibrator


def interval_diagnostics(intervals, trues, bin_key, alpha=0.1, n_bins=10):
    """Width/coverage/adaptivity summary for one method's [N,2] intervals.

    `bin_key` is the per-point quantity whose quantile bins conditional coverage is measured
    over (epistemic variance or total variance). Returns a dict with:
      cov         marginal coverage
      width       mean interval width, and `median_width`, and their ratio as a skew flag
      cov_gap     mean |bin coverage - (1-alpha)| across the bins -- the headline adaptivity
                  number; a method can hit 90% marginally while badly missing the hard bins
      worst_bin   the least-covered bin
      score       Winkler interval score (U-L) + (2/alpha)[(L-y)+ + (y-U)+], a proper scoring
                  rule that charges width and misses together, so a method cannot win it by
                  shrinking intervals where it already under-covers
    """
    lower, upper = intervals[:, 0], intervals[:, 1]
    covered = (trues >= lower) & (trues <= upper)
    widths = upper - lower

    edges = np.quantile(bin_key, np.linspace(0.0, 1.0, n_bins + 1))
    edges[-1] += 1e-9
    b = np.clip(np.digitize(bin_key, edges) - 1, 0, n_bins - 1)
    bin_cov = np.array([covered[b == i].mean() if np.any(b == i) else np.nan
                        for i in range(n_bins)])
    valid = bin_cov[~np.isnan(bin_cov)]

    score = (widths + (2.0 / alpha) * (np.clip(lower - trues, 0, None)
                                       + np.clip(trues - upper, 0, None)))
    median_width = float(np.median(widths))
    mean_width = float(np.mean(widths))
    return {
        'cov': float(covered.mean()),
        'width': mean_width,
        'median_width': median_width,
        'skew': mean_width / median_width if median_width > 0 else float('nan'),
        'cov_gap': float(np.abs(valid - (1 - alpha)).mean()) if valid.size else float('nan'),
        'worst_bin': float(valid.min()) if valid.size else float('nan'),
        'score': float(score.mean()),
        'bin_cov': bin_cov,
    }


def _fmt_diag(name, d):
    return (f"{name:<34s} width={d['width']:.4f} med={d['median_width']:.4f} "
            f"skew={d['skew']:.2f} cov={d['cov']:.4f} CovGap={d['cov_gap']:.4f} "
            f"worst={d['worst_bin']:.3f} IntScore={d['score']:.4f}")


def _curve_significance(mean_curve, std_curve, n_repeats):
    """Which grid points are statistically indistinguishable from the curve's minimum.

    Returns (best_key, n_within_se, n_within_sd, se). A tuned ratio whose minimum has many
    grid points inside one standard error is not a signal -- it is the split noise.
    """
    keys = sorted(mean_curve.keys())
    vals = np.array([mean_curve[k] for k in keys])
    j = int(np.argmin(vals))
    if not std_curve or n_repeats in (None, 0):
        return keys[j], None, None, None
    sd = std_curve.get(keys[j], float('nan'))
    se = sd / np.sqrt(n_repeats)
    return (keys[j], int(np.sum(vals < vals[j] + se)), int(np.sum(vals < vals[j] + sd)), se)


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

        if getattr(self.args, 'use_quantile_loss', False):
            # Aggregates per-expert (lower, upper) predictions via the gate FIRST, then computes
            # one pinball loss on the aggregate -- unlike the MoE branches below (which weight
            # per-expert *losses*), because QuantileLoss.forward() always reduces to a scalar
            # and has no per-sample mode to multiply by gate weights before summing.
            if self.args.num_experts > 1:
                expert_out, _, weights = self.model(batch_x)
                avg_weight = weights.squeeze(-1).mean(dim=0)
                load_balance_loss = getattr(self.args, 'load_balance_weight', 0.05) * self.args.num_experts * torch.sum(avg_weight * avg_weight)
                pred = torch.sum(expert_out * weights, dim=1)
                loss = QuantileLoss(quantiles=[0.05, 0.95])(pred, batch_y) + load_balance_loss
            else:
                pred = self.model(batch_x)
                loss = QuantileLoss(quantiles=[0.05, 0.95])(pred, batch_y)
            return loss

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
            load_balance_loss = getattr(self.args, 'load_balance_weight', 0.05) * self.args.num_experts * torch.sum(avg_weight * avg_weight)

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
                var = torch.clamp(var, min=0.01)
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

                    if getattr(self.args, 'use_quantile_loss', False):
                        # pred is [B,2] = (lower, upper); derive a proxy point/std so this
                        # shared method stays usable by test() without shape errors.
                        pred_lower, pred_upper = pred[:, 0:1], pred[:, 1:2]
                        pred = (pred_lower + pred_upper) / 2
                        std_total = (pred_upper - pred_lower) / 2
                        std_aleat = std_total
                        std_epist = torch.zeros_like(std_total)
                    elif self.args.prob_expert:
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
                    if getattr(self.args, 'use_quantile_loss', False):
                        pred = self.model(batch_x)  # [B,2] = (lower, upper)
                        pred_lower, pred_upper = pred[:, 0:1], pred[:, 1:2]
                        pred = (pred_lower + pred_upper) / 2
                        std_total = (pred_upper - pred_lower) / 2
                        std_aleat = std_total
                        std_epist = torch.zeros_like(std_total)
                    elif self.args.prob_expert:
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

    def _collect_quantile_predictions(self, dataloader):
        """
        Raw (lower, upper) quantile-head outputs for CQR conformalization, used only by
        calibrate_cqr. Requires args.use_quantile_loss=True.
        """
        assert getattr(self.args, 'use_quantile_loss', False), \
            "_collect_quantile_predictions requires --use_quantile_loss"
        self.model.eval()
        all_lower, all_upper, all_trues = [], [], []

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(dataloader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if self.args.num_experts > 1:
                    expert_out, _, weights = self.model(batch_x)
                    pred = torch.sum(expert_out * weights, dim=1)  # [B,2] = (lower, upper)
                else:
                    pred = self.model(batch_x)  # [B,2] = (lower, upper)

                all_lower.append(pred[:, 0].detach().cpu())
                all_upper.append(pred[:, 1].detach().cpu())
                all_trues.append(batch_y.detach().cpu())

        return torch.cat(all_lower), torch.cat(all_upper), torch.cat(all_trues)

    def _collect_mog_predictions(self, dataloader):
        """
        Raw per-expert (mu_k, sigma_k, pi_k) tensors for the MoG-HPD calibrator, used only by
        calibrate_mog_hpd. Requires args.prob_expert=True (per-expert variance only exists for
        probabilistic experts) and is incompatible with args.use_quantile_loss.
        """
        assert getattr(self.args, 'prob_expert', False), \
            "_collect_mog_predictions requires --prob_expert"
        assert not getattr(self.args, 'use_quantile_loss', False), \
            "_collect_mog_predictions is incompatible with --use_quantile_loss"
        self.model.eval()
        all_mu, all_sigma, all_pi, all_trues = [], [], [], []

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if self.args.num_experts > 1:
                    expert_out, expert_unc, weights = self.model(batch_x)  # [B,K,1] each
                    mu, var, pi = expert_out.squeeze(-1), expert_unc.squeeze(-1), weights.squeeze(-1)
                else:
                    mean, var_ = self.model(batch_x)  # SimpleMLP path: [B,1] each
                    mu, var, pi = mean, var_, torch.ones_like(mean)

                all_mu.append(mu.detach().cpu())
                # sigma = sqrt(var); the model's softplus(...)+1e-6 head already guarantees
                # var>0, the extra 1e-12 here is just defensive against float underflow.
                all_sigma.append(torch.sqrt(var + 1e-12).detach().cpu())
                all_pi.append(pi.detach().cpu())
                all_trues.append(batch_y.detach().cpu())

        mu, sigma, pi, trues = (torch.cat(all_mu), torch.cat(all_sigma),
                                torch.cat(all_pi), torch.cat(all_trues))

        # Structured summary of the raw mixture parameters every MoG-family calibrator is handed.
        # One line, not one per point, so the logs/tabular/*.log regex parsing is unaffected.
        def _mmm(t):
            return f"{t.min().item():.4g}/{t.median().item():.4g}/{t.max().item():.4g}"

        pi_safe = pi.clamp_min(1e-12)
        gate_entropy = -(pi_safe * pi_safe.log()).sum(dim=1).mean().item()
        print(f"[STA-HPD] mog_params: n={mu.shape[0]} K={mu.shape[1]} "
              f"mu_min_med_max={_mmm(mu)} sigma_min_med_max={_mmm(sigma)} pi_min_med_max={_mmm(pi)} "
              f"n_sigma_below_1e-4={int((sigma < 1e-4).sum().item())} "
              f"gate_entropy_mean={gate_entropy:.4f} "
              # Near-one-hot routing flattens the mixture toward a single Gaussian (and is the
              # same condition that starves MoECP's similarity weighting -- see MoECP_Calibrator).
              f"frac_top_gate_above_0.99={(pi.max(dim=1).values > 0.99).float().mean().item():.4f}")

        return mu, sigma, pi, trues

    def _collect_expert_predictions(self, dataloader):
        """
        Raw per-expert (mu_k, pi_k) tensors for the MoCE calibrator, used only by calibrate_moce.
        Unlike _collect_mog_predictions this does not require args.prob_expert (no per-expert
        variance is needed), so it applies uniformly to ClassicMoE, MOG, and MOGU. Incompatible
        with args.use_quantile_loss (no single per-expert point mean in that case).
        """
        assert not getattr(self.args, 'use_quantile_loss', False), \
            "_collect_expert_predictions is incompatible with --use_quantile_loss"
        self.model.eval()
        all_mu, all_pi, all_trues = [], [], []

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if self.args.num_experts > 1:
                    expert_out, _, weights = self.model(batch_x)  # [B,K,1] each
                    mu, pi = expert_out.squeeze(-1), weights.squeeze(-1)
                else:
                    mean = self.model(batch_x)
                    if self.args.prob_expert:
                        mean = mean[0]
                    mu, pi = mean, torch.ones_like(mean)

                all_mu.append(mu.detach().cpu())
                all_pi.append(pi.detach().cpu())
                all_trues.append(batch_y.detach().cpu())

        return torch.cat(all_mu), torch.cat(all_pi), torch.cat(all_trues)

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

        set_seed(self.args.seed)

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

    def calibrate_adaptive_variance(self, setting):
        path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
        self.model.eval()

        cal_data, cal_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        cal_preds, cal_trues, cal_weights, cal_stds_total, cal_stds_aleat, cal_stds_epist = self._collect_predictions(cal_loader)
        test_preds, test_trues, test_weights, test_stds_total, test_stds_aleat, test_stds_epist = self._collect_predictions(test_loader)

        calibrator = AdaptiveVarianceCalibrator(alpha=0.1)
        calibrator.fit(cal_preds, cal_trues, cal_stds_aleat, cal_stds_epist)
        intervals = calibrator.predict(test_preds, test_stds_aleat, test_stds_epist)

        intervals = intervals.numpy()
        test_trues = test_trues.squeeze().numpy()

        coverage = np.mean((test_trues >= intervals[:, 0]) & (test_trues <= intervals[:, 1]))
        width = np.mean(intervals[:, 1] - intervals[:, 0])
        median_width = np.median(intervals[:, 1] - intervals[:, 0])

        print(f"\nAdaptive Variance-Ratio Results (Learned r = {calibrator.r:.4f}, q^2 = {calibrator.q_sq:.4f}, "
              f"n_tune={calibrator.n_tune_}, n_calib={calibrator.n_calib_}, n_repeats={calibrator.n_repeats_}):")
        print(f"Coverage: {coverage:.4f}")
        print(f"Avg Width: {width:.4f}")
        print(f"Median Width: {median_width:.4f}")

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open("result_calibration_adaptive_variance.txt", 'a') as f:
            f.write(f"{setting} (Adaptive Variance-Ratio)\n")
            f.write(f"Coverage: {coverage:.4f}, Width: {width:.4f}, Median Width: {median_width:.4f}, "
                    f"r: {calibrator.r:.4f}, q_sq: {calibrator.q_sq:.4f}, n_repeats: {calibrator.n_repeats_}\n\n")

        np.save(folder_path + 'intervals_adaptive_variance.npy', intervals)

        try:
            eps = 1e-8

            def _safe_corr(x, y, corr_fn):
                if np.std(x) < eps or np.std(y) < eps:
                    return float('nan'), float('nan')
                r, p = corr_fn(x, y)
                return float(r), float(p)

            diag_preds = test_preds.squeeze().numpy()
            diag_resid_sq = (test_trues - diag_preds) ** 2
            diag_aleat_var = test_stds_aleat.squeeze().numpy() ** 2
            diag_epist_var = test_stds_epist.squeeze().numpy() ** 2
            diag_total_var = diag_aleat_var + diag_epist_var

            print("\n" + "="*40)
            print("--- Adaptive-Variance Diagnostics ---")

            # [1] Does the model's own variance actually track the realized error?
            pear_aleat, _ = _safe_corr(diag_aleat_var, diag_resid_sq, scipy.stats.pearsonr)
            pear_epist, _ = _safe_corr(diag_epist_var, diag_resid_sq, scipy.stats.pearsonr)
            pear_total, _ = _safe_corr(diag_total_var, diag_resid_sq, scipy.stats.pearsonr)
            spear_aleat, _ = _safe_corr(diag_aleat_var, diag_resid_sq, scipy.stats.spearmanr)
            spear_epist, _ = _safe_corr(diag_epist_var, diag_resid_sq, scipy.stats.spearmanr)
            spear_total, _ = _safe_corr(diag_total_var, diag_resid_sq, scipy.stats.spearmanr)
            print(f"[1] Variance-vs-error correlation (test set, N={len(diag_resid_sq)}), resid_sq=(y_true-pred)^2:")
            print(f"    Pearson  r: aleat={pear_aleat:.4f}, epist={pear_epist:.4f}, total={pear_total:.4f}")
            print(f"    Spearman r: aleat={spear_aleat:.4f}, epist={spear_epist:.4f}, total={spear_total:.4f}")

            # [2] Is the model's variance well-scaled on average, or systematically over/under-dispersed?
            diag_scale_ratio = float(np.mean(diag_resid_sq) / (np.mean(diag_total_var) + eps))
            print(f"[2] Variance-scale check: mean(resid_sq)/mean(total_var) = {diag_scale_ratio:.4f} "
                  f"(~1.0 = well-scaled; >1 = underdispersed model variance -> forces q_sq>1; <1 = overdispersed)")

            # [3] Shape of the r-search: real interior trade-off, or a degenerate boundary solution?
            diag_r_sorted = sorted(calibrator.tuning_widths_.keys())
            diag_widths_sorted = [calibrator.tuning_widths_[r] for r in diag_r_sorted]
            diag_is_boundary = (calibrator.r == diag_r_sorted[0]) or (calibrator.r == diag_r_sorted[-1])
            diag_nested_used = calibrator.n_tune_ is not None
            print(f"[3] r-search curve ({len(diag_r_sorted)} grid points, "
                  f"{'nested tune/calib split' if diag_nested_used else 'SINGLE-FOLD (in-sample, may overstate any interior minimum)'}): "
                  f"chosen r={calibrator.r:.4f} ({'BOUNDARY' if diag_is_boundary else 'interior'} solution), "
                  f"width range [{min(diag_widths_sorted):.4f}, {max(diag_widths_sorted):.4f}]")
            diag_curve_str = ", ".join(f"{r:.4f}:{w:.4f}" for r, w in zip(diag_r_sorted, diag_widths_sorted))
            print(f"    full curve (r:tune_width): {diag_curve_str}")

            # [4] Same cal/test split, what would CP-VS(total) and MoECP have produced?
            ref_cpvs = CP_VS_Static_Calibrator(alpha=0.1)
            ref_cpvs.fit(cal_preds, cal_trues, cal_stds_total)
            ref_cpvs_intervals = ref_cpvs.predict(test_preds, test_stds_total).numpy()
            ref_cpvs_width = float(np.mean(ref_cpvs_intervals[:, 1] - ref_cpvs_intervals[:, 0]))
            ref_cpvs_coverage = float(np.mean((test_trues >= ref_cpvs_intervals[:, 0]) & (test_trues <= ref_cpvs_intervals[:, 1])))
            print(f"[4a] Same-split reference, CP-VS(total): coverage={ref_cpvs_coverage:.4f}, avg width={ref_cpvs_width:.4f}  "
                  f"| Adaptive-Variance (this run): coverage={coverage:.4f}, avg width={width:.4f}")

            diag_moecp_pairs = len(cal_preds) * len(test_preds)
            diag_moecp_max_pairs = 5_000_000
            if diag_moecp_pairs <= diag_moecp_max_pairs:
                try:
                    ref_moecp = MoECP_Calibrator(alpha=0.1, temperature=self.args.tau)
                    ref_moecp.fit(cal_preds, cal_trues, cal_weights)
                    ref_moecp_intervals = ref_moecp.predict(test_preds, test_weights).numpy()
                    ref_moecp_width = float(np.mean(ref_moecp_intervals[:, 1] - ref_moecp_intervals[:, 0]))
                    ref_moecp_coverage = float(np.mean((test_trues >= ref_moecp_intervals[:, 0]) & (test_trues <= ref_moecp_intervals[:, 1])))
                    print(f"[4b] Same-split reference, MoECP: coverage={ref_moecp_coverage:.4f}, avg width={ref_moecp_width:.4f}  "
                          f"| Adaptive-Variance (this run): coverage={coverage:.4f}, avg width={width:.4f}")
                except Exception as e:
                    ref_moecp_width, ref_moecp_coverage = float('nan'), float('nan')
                    print(f"[4b] Same-split reference, MoECP: skipped due to error: {e}")
            else:
                ref_moecp_width, ref_moecp_coverage = float('nan'), float('nan')
                print(f"[4b] Same-split reference, MoECP: skipped (n_cal*n_test={diag_moecp_pairs} exceeds diagnostic cap {diag_moecp_max_pairs})")

            print("="*40 + "\n")

            with open("result_calibration_adaptive_variance.txt", 'a') as f:
                f.write(f"{setting} (Adaptive Variance-Ratio Diagnostics)\n")
                f.write(f"corr_pearson(aleat,epist,total)=({pear_aleat:.4f},{pear_epist:.4f},{pear_total:.4f}), "
                        f"corr_spearman(aleat,epist,total)=({spear_aleat:.4f},{spear_epist:.4f},{spear_total:.4f}), "
                        f"scale_ratio={diag_scale_ratio:.4f}, nested_used={diag_nested_used}, "
                        f"r_boundary={diag_is_boundary}, chosen_r={calibrator.r:.4f}, "
                        f"ref_cpvs_total: coverage={ref_cpvs_coverage:.4f} width={ref_cpvs_width:.4f}, "
                        f"ref_moecp: coverage={ref_moecp_coverage:.4f} width={ref_moecp_width:.4f}\n")
                f.write(f"tuning_curve (r:tune_width): {diag_curve_str}\n\n")
        except Exception as e:
            print(f"[Adaptive-Variance Diagnostics] skipped due to error: {e}")

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

    def calibrate_cqr(self, setting):
        path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
        self.model.eval()

        cal_data, cal_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        cal_lower, cal_upper, cal_trues = self._collect_quantile_predictions(cal_loader)
        test_lower, test_upper, test_trues = self._collect_quantile_predictions(test_loader)

        calibrator = CQR_Calibrator(alpha=0.1)
        calibrator.fit(cal_lower, cal_upper, cal_trues)
        intervals = calibrator.predict(test_lower, test_upper)

        test_trues = test_trues.squeeze().numpy()

        coverage = np.mean((test_trues >= intervals[:, 0]) & (test_trues <= intervals[:, 1]))
        width = np.mean(intervals[:, 1] - intervals[:, 0])
        median_width = np.median(intervals[:, 1] - intervals[:, 0])

        print(f"\nCQR Results:")
        print(f"Coverage: {coverage:.4f}")
        print(f"Avg Width: {width:.4f}")
        print(f"Median Width: {median_width:.4f}")

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open("result_calibration_cqr.txt", 'a') as f:
            f.write(f"{setting} (CQR)\n")
            f.write(f"Coverage: {coverage:.4f}, Width: {width:.4f}, Median Width: {median_width:.4f}\n\n")

        np.save(folder_path + 'intervals_cqr.npy', intervals)
        return coverage, width

    def calibrate_mog_hpd(self, setting):
        path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
        self.model.eval()

        cal_data, cal_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        cal_mu, cal_sigma, cal_pi, cal_trues = self._collect_mog_predictions(cal_loader)
        test_mu, test_sigma, test_pi, test_trues = self._collect_mog_predictions(test_loader)

        calibrator = MoG_HPD_Calibrator(alpha=0.1)
        calibrator.fit(cal_trues, cal_mu, cal_sigma, cal_pi)
        intervals = calibrator.predict(test_mu, test_sigma, test_pi)  # list of [m_i,2] arrays

        test_trues_np = test_trues.squeeze().numpy()

        widths = np.array([seg[:, 1].sum() - seg[:, 0].sum() if len(seg) else 0.0 for seg in intervals])
        covered = np.array([bool(len(seg)) and bool(np.any((seg[:, 0] <= y) & (y <= seg[:, 1])))
                             for y, seg in zip(test_trues_np, intervals)])
        n_segments = np.array([len(seg) for seg in intervals])

        coverage = covered.mean()
        width = widths.mean()
        median_width = np.median(widths)
        frac_multi = np.mean(n_segments > 1)
        frac_empty = np.mean(n_segments == 0)
        avg_segments = n_segments.mean()

        print(f"\nMoG-HPD Results (t_hat={calibrator.t_hat:.4f}, tau={calibrator.tau_:.6g}, "
              f"avg segments={avg_segments:.3f}, frac multi-segment={frac_multi:.4f}, "
              f"frac empty={frac_empty:.4f}):")
        print(f"Coverage: {coverage:.4f}")
        print(f"Avg Width: {width:.4f}")
        print(f"Median Width: {median_width:.4f}")

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open("result_calibration_mog_hpd.txt", 'a') as f:
            f.write(f"{setting} (MoG-HPD)\n")
            f.write(f"Coverage: {coverage:.4f}, Width: {width:.4f}, Median Width: {median_width:.4f}, "
                    f"t_hat: {calibrator.t_hat:.4f}, avg_segments: {avg_segments:.3f}, "
                    f"frac_multi: {frac_multi:.4f}, frac_empty: {frac_empty:.4f}\n\n")

        # Ragged per-point segment counts -> pickled object array (unlike the other methods'
        # plain [N,2] float .npy files, this needs allow_pickle=True on load).
        np.save(folder_path + 'intervals_mog_hpd.npy', np.array(intervals, dtype=object), allow_pickle=True)
        return coverage, width

    def calibrate_sta_hpd(self, setting):
        path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
        self.model.eval()

        cal_data, cal_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        cal_mu, cal_sigma, cal_pi, cal_trues = self._collect_mog_predictions(cal_loader)
        test_mu, test_sigma, test_pi, test_trues = self._collect_mog_predictions(test_loader)

        calibrator = STAHPDCalibrator(alpha=0.1)
        calibrator.fit(cal_trues, cal_mu, cal_sigma, cal_pi)
        intervals = calibrator.predict(test_mu, test_sigma, test_pi)  # list of [m_i,2] arrays

        test_trues_np = test_trues.squeeze().numpy()

        widths = np.array([seg[:, 1].sum() - seg[:, 0].sum() if len(seg) else 0.0 for seg in intervals])
        covered = np.array([bool(len(seg)) and bool(np.any((seg[:, 0] <= y) & (y <= seg[:, 1])))
                             for y, seg in zip(test_trues_np, intervals)])
        n_segments = np.array([len(seg) for seg in intervals])

        coverage = covered.mean()
        width = widths.mean()
        median_width = np.median(widths)
        frac_multi = np.mean(n_segments > 1)
        frac_empty = np.mean(n_segments == 0)
        avg_segments = n_segments.mean()

        # Header is "STA-HPD Results" -- deliberately NOT containing "MoG-HPD", so the
        # run_tabular_regression.sh parser's anchored '\nMoG-HPD Results' pattern cannot
        # substring-match this method (and vice versa).
        print(f"\nSTA-HPD Results (c*={calibrator.c_:.4f}, theta*={calibrator.theta_:.4f}, "
              f"tune width@best={calibrator.tune_width_at_best_:.4f}, t_hat={calibrator.t_hat:.4f}, "
              f"G={calibrator.G_:.6g}, avg segments={avg_segments:.3f}, "
              f"frac multi-segment={frac_multi:.4f}, frac empty={frac_empty:.4f}, "
              f"n_tune={calibrator.n_tune_}, n_calib={calibrator.n_calib_}, "
              f"n_repeats={calibrator.n_repeats_}, single_fold={calibrator.single_fold_used_}):")
        print(f"Coverage: {coverage:.4f}")
        print(f"Avg Width: {width:.4f}")
        print(f"Median Width: {median_width:.4f}")

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open("result_calibration_sta_hpd.txt", 'a') as f:
            f.write(f"{setting} (STA-HPD)\n")
            f.write(f"Coverage: {coverage:.4f}, Width: {width:.4f}, Median Width: {median_width:.4f}, "
                    f"c: {calibrator.c_:.4f}, theta: {calibrator.theta_:.4f}, "
                    f"t_hat: {calibrator.t_hat:.4f}, avg_segments: {avg_segments:.3f}, "
                    f"frac_multi: {frac_multi:.4f}, frac_empty: {frac_empty:.4f}, "
                    f"single_fold: {calibrator.single_fold_used_}\n\n")

        # Ragged per-point segment counts -> pickled object array (same convention as MoG-HPD).
        np.save(folder_path + 'intervals_sta_hpd.npy', np.array(intervals, dtype=object), allow_pickle=True)

        # Diagnostics: the (c, theta) tuning surface, the vote spread across splits, and what
        # plain MoG-HPD -- the (c=1, theta=0) cell of that same surface, and the method this must
        # beat -- produced on the identical cal/test split.
        try:
            diag_keys = sorted(calibrator.tuning_widths_.keys())
            diag_best = (calibrator.c_, calibrator.theta_)
            diag_c_edge = calibrator.c_ in (calibrator.c_grid[0], calibrator.c_grid[-1])
            diag_th_edge = calibrator.theta_ in (calibrator.theta_grid[0], calibrator.theta_grid[-1])
            diag_is_boundary = diag_c_edge or diag_th_edge
            diag_widths = [calibrator.tuning_widths_[k] for k in diag_keys]
            diag_surface_str = ", ".join(f"({c:g},{th:g}):{calibrator.tuning_widths_[(c, th)]:.4f}"
                                          for c, th in diag_keys)
            diag_votes = sorted(calibrator.vote_counts_.items(), key=lambda kv: -kv[1])
            diag_votes_str = ", ".join(f"({c:g},{th:g}):{v}" for (c, th), v in diag_votes)

            ref_hpd = MoG_HPD_Calibrator(alpha=0.1)
            ref_hpd.fit(cal_trues, cal_mu, cal_sigma, cal_pi)
            ref_intervals = ref_hpd.predict(test_mu, test_sigma, test_pi)
            ref_widths = np.array([seg[:, 1].sum() - seg[:, 0].sum() if len(seg) else 0.0
                                   for seg in ref_intervals])
            ref_covered = np.array([bool(len(seg)) and bool(np.any((seg[:, 0] <= y) & (y <= seg[:, 1])))
                                     for y, seg in zip(test_trues_np, ref_intervals)])
            ref_width = float(ref_widths.mean())
            ref_coverage = float(ref_covered.mean())

            print("\n" + "="*40)
            print("--- STA-HPD Diagnostics ---")
            print(f"[1] (c,theta) surface ({len(diag_keys)} cells, "
                  f"{'SINGLE-FOLD (in-sample, may overstate any interior minimum)' if calibrator.single_fold_used_ else f'{calibrator.n_repeats_} tune/calib splits'}): "
                  f"chosen (c*,theta*)=({calibrator.c_:.4f},{calibrator.theta_:.4f}) "
                  f"({'BOUNDARY' if diag_is_boundary else 'interior'} solution), "
                  f"tune width range [{min(diag_widths):.4f}, {max(diag_widths):.4f}]")
            print(f"    vote spread (cell:n_splits): {diag_votes_str}")
            print(f"    full surface ((c,theta):tune_width): {diag_surface_str}")
            print(f"[2] Same-split reference, plain MoG-HPD (= the (c=1,theta=0) cell): "
                  f"coverage={ref_coverage:.4f}, avg width={ref_width:.4f}  | STA-HPD (this run): "
                  f"coverage={coverage:.4f}, avg width={width:.4f} "
                  f"({100.0 * (width - ref_width) / ref_width:+.1f}%)")
            print("="*40 + "\n")

            with open("result_calibration_sta_hpd.txt", 'a') as f:
                f.write(f"{setting} (STA-HPD Diagnostics)\n")
                f.write(f"boundary={diag_is_boundary}, chosen_c={calibrator.c_:.4f}, "
                        f"chosen_theta={calibrator.theta_:.4f}, "
                        f"single_fold={calibrator.single_fold_used_}, "
                        f"ref_mog_hpd: coverage={ref_coverage:.4f} width={ref_width:.4f} | "
                        f"sta_hpd: coverage={coverage:.4f} width={width:.4f}\n")
                f.write(f"votes (c,theta):n_splits: {diag_votes_str}\n")
                f.write(f"tuning_surface ((c,theta):tune_width): {diag_surface_str}\n\n")
        except Exception as e:
            print(f"[STA-HPD Diagnostics] skipped due to error: {e}")

        return coverage, width

    def calibrate_seta_hpd(self, setting):
        """
        SETA-HPD: STA-HPD extended with the MoG between-component (epistemic) variance-decomposition
        knob rho. Runs independently of (and alongside) calibrate_sta_hpd -- separate calibrator,
        separate result file. rho only has effect for K>1; for single-expert models it coincides with
        STA-HPD.
        """
        path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
        self.model.eval()

        cal_data, cal_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        cal_mu, cal_sigma, cal_pi, cal_trues = self._collect_mog_predictions(cal_loader)
        test_mu, test_sigma, test_pi, test_trues = self._collect_mog_predictions(test_loader)

        # mog_decompose defaults to True in SETAHPDCalibrator (rho on); pass through an optional
        # override so the epistemic knob can be disabled to recover STA-HPD if desired.
        mog_decompose = getattr(self.args, 'seta_hpd_mog_decompose', True)
        calibrator = SETAHPDCalibrator(alpha=0.1, mog_decompose=mog_decompose)
        calibrator.fit(cal_trues, cal_mu, cal_sigma, cal_pi)
        intervals = calibrator.predict(test_mu, test_sigma, test_pi)  # list of [m_i,2] arrays

        test_trues_np = test_trues.squeeze().numpy()

        widths = np.array([seg[:, 1].sum() - seg[:, 0].sum() if len(seg) else 0.0 for seg in intervals])
        covered = np.array([bool(len(seg)) and bool(np.any((seg[:, 0] <= y) & (y <= seg[:, 1])))
                             for y, seg in zip(test_trues_np, intervals)])
        n_segments = np.array([len(seg) for seg in intervals])

        coverage = covered.mean()
        width = widths.mean()
        median_width = np.median(widths)
        frac_multi = np.mean(n_segments > 1)
        frac_empty = np.mean(n_segments == 0)
        avg_segments = n_segments.mean()

        # Header is "SETA-HPD Results" -- deliberately NOT containing "STA-HPD" as a contiguous
        # substring ("SETA" is S-E-T-A, not S-T-A), so the run_tabular_regression.sh parser's
        # 'STA-HPD Results' / 'MoG-HPD Results' patterns cannot substring-match this method.
        print(f"\nSETA-HPD Results (c*={calibrator.c_:.4f}, rho*={calibrator.rho_:.4f}, "
              f"theta*={calibrator.theta_:.4f}, mog_decompose={calibrator.mog_decompose}, "
              f"tune width@best={calibrator.tune_width_at_best_:.4f}, t_hat={calibrator.t_hat:.4f}, "
              f"G={calibrator.G_:.6g}, avg segments={avg_segments:.3f}, "
              f"frac multi-segment={frac_multi:.4f}, frac empty={frac_empty:.4f}, "
              f"n_tune={calibrator.n_tune_}, n_calib={calibrator.n_calib_}, "
              f"n_repeats={calibrator.n_repeats_}, single_fold={calibrator.single_fold_used_}):")
        print(f"Coverage: {coverage:.4f}")
        print(f"Avg Width: {width:.4f}")
        print(f"Median Width: {median_width:.4f}")

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open("result_calibration_seta_hpd.txt", 'a') as f:
            f.write(f"{setting} (SETA-HPD)\n")
            f.write(f"Coverage: {coverage:.4f}, Width: {width:.4f}, Median Width: {median_width:.4f}, "
                    f"c: {calibrator.c_:.4f}, rho: {calibrator.rho_:.4f}, "
                    f"theta: {calibrator.theta_:.4f}, mog_decompose: {calibrator.mog_decompose}, "
                    f"t_hat: {calibrator.t_hat:.4f}, avg_segments: {avg_segments:.3f}, "
                    f"frac_multi: {frac_multi:.4f}, frac_empty: {frac_empty:.4f}, "
                    f"single_fold: {calibrator.single_fold_used_}\n\n")

        np.save(folder_path + 'intervals_seta_hpd.npy', np.array(intervals, dtype=object), allow_pickle=True)

        # Diagnostics: the (c, rho, theta) tuning surface, a compact rho-marginal, the vote spread,
        # and both plain MoG-HPD and STA-HPD (the rho=1 slice) on the identical cal/test split -- the
        # two methods SETA-HPD must beat.
        try:
            diag_keys = sorted(calibrator.tuning_widths_.keys())
            diag_c_edge = calibrator.c_ in (calibrator.c_grid[0], calibrator.c_grid[-1])
            diag_rho_edge = calibrator.mog_decompose and \
                calibrator.rho_ in (calibrator.rho_grid[0], calibrator.rho_grid[-1])
            diag_th_edge = calibrator.theta_ in (calibrator.theta_grid[0], calibrator.theta_grid[-1])
            diag_is_boundary = diag_c_edge or diag_rho_edge or diag_th_edge
            diag_widths = [calibrator.tuning_widths_[k] for k in diag_keys]
            diag_surface_str = ", ".join(f"({c:g},{rho:g},{th:g}):{calibrator.tuning_widths_[(c, rho, th)]:.4f}"
                                          for c, rho, th in diag_keys)
            diag_rho_marg = {}
            for (c, rho, th), w in calibrator.tuning_widths_.items():
                diag_rho_marg[rho] = min(diag_rho_marg.get(rho, np.inf), w)
            diag_rho_marg_str = ", ".join(f"{rho:g}:{diag_rho_marg[rho]:.4f}"
                                          for rho in sorted(diag_rho_marg))
            diag_votes = sorted(calibrator.vote_counts_.items(), key=lambda kv: -kv[1])
            diag_votes_str = ", ".join(f"({c:g},{rho:g},{th:g}):{v}" for (c, rho, th), v in diag_votes)

            ref_hpd = MoG_HPD_Calibrator(alpha=0.1)
            ref_hpd.fit(cal_trues, cal_mu, cal_sigma, cal_pi)
            ref_intervals = ref_hpd.predict(test_mu, test_sigma, test_pi)
            ref_widths = np.array([seg[:, 1].sum() - seg[:, 0].sum() if len(seg) else 0.0
                                   for seg in ref_intervals])
            ref_covered = np.array([bool(len(seg)) and bool(np.any((seg[:, 0] <= y) & (y <= seg[:, 1])))
                                     for y, seg in zip(test_trues_np, ref_intervals)])
            ref_width = float(ref_widths.mean())
            ref_coverage = float(ref_covered.mean())

            # STA-HPD (rho=1 slice) on the same split, i.e. what SETA-HPD's between-component knob
            # adds on top of the existing two-parameter method.
            ref_sta = STAHPDCalibrator(alpha=0.1, verbose=False)
            ref_sta.fit(cal_trues, cal_mu, cal_sigma, cal_pi)
            sta_intervals = ref_sta.predict(test_mu, test_sigma, test_pi)
            sta_widths = np.array([seg[:, 1].sum() - seg[:, 0].sum() if len(seg) else 0.0
                                   for seg in sta_intervals])
            sta_covered = np.array([bool(len(seg)) and bool(np.any((seg[:, 0] <= y) & (y <= seg[:, 1])))
                                     for y, seg in zip(test_trues_np, sta_intervals)])
            sta_width = float(sta_widths.mean())
            sta_coverage = float(sta_covered.mean())

            print("\n" + "="*40)
            print("--- SETA-HPD Diagnostics ---")
            print(f"[1] (c,rho,theta) surface ({len(diag_keys)} cells, mog_decompose={calibrator.mog_decompose}, "
                  f"{'SINGLE-FOLD (in-sample, may overstate any interior minimum)' if calibrator.single_fold_used_ else f'{calibrator.n_repeats_} tune/calib splits'}): "
                  f"chosen (c*,rho*,theta*)=({calibrator.c_:.4f},{calibrator.rho_:.4f},{calibrator.theta_:.4f}) "
                  f"({'BOUNDARY' if diag_is_boundary else 'interior'} solution), "
                  f"tune width range [{min(diag_widths):.4f}, {max(diag_widths):.4f}]")
            print(f"    vote spread (cell:n_splits): {diag_votes_str}")
            print(f"    rho-marginal (best tune width over c,theta | rho): {diag_rho_marg_str}")
            print(f"    full surface ((c,rho,theta):tune_width): {diag_surface_str}")
            print(f"[2] Same-split references: MoG-HPD coverage={ref_coverage:.4f} width={ref_width:.4f} "
                  f"({100.0 * (width - ref_width) / ref_width:+.1f}% vs SETA); "
                  f"STA-HPD (rho=1) coverage={sta_coverage:.4f} width={sta_width:.4f} "
                  f"({100.0 * (width - sta_width) / sta_width:+.1f}% vs SETA)  | "
                  f"SETA-HPD (this run): coverage={coverage:.4f}, avg width={width:.4f}")
            print("="*40 + "\n")

            with open("result_calibration_seta_hpd.txt", 'a') as f:
                f.write(f"{setting} (SETA-HPD Diagnostics)\n")
                f.write(f"boundary={diag_is_boundary}, chosen_c={calibrator.c_:.4f}, "
                        f"chosen_rho={calibrator.rho_:.4f}, chosen_theta={calibrator.theta_:.4f}, "
                        f"mog_decompose={calibrator.mog_decompose}, "
                        f"single_fold={calibrator.single_fold_used_}, "
                        f"ref_mog_hpd: coverage={ref_coverage:.4f} width={ref_width:.4f} | "
                        f"ref_sta_hpd: coverage={sta_coverage:.4f} width={sta_width:.4f} | "
                        f"seta_hpd: coverage={coverage:.4f} width={width:.4f}\n")
                f.write(f"votes (c,rho,theta):n_splits: {diag_votes_str}\n")
                f.write(f"rho_marginal (rho:best_tune_width): {diag_rho_marg_str}\n")
                f.write(f"tuning_surface ((c,rho,theta):tune_width): {diag_surface_str}\n\n")
        except Exception as e:
            print(f"[SETA-HPD Diagnostics] skipped due to error: {e}")

        return coverage, width

    def calibrate_meld_hpd(self, setting):
        """
        MELD-HPD: same (c, rho, theta) surrogate family as SETA-HPD, but the shape exponents
        (c, rho) are selected by MAXIMIZING mean calibration log predictive density (a proper
        scoring rule -> best density match -> Neyman-Pearson-narrowest HPD) instead of by argmin
        tune-fold width, while theta is still chosen by width. Runs independently of (and
        alongside) calibrate_sta_hpd / calibrate_seta_hpd -- separate calibrator, separate result
        file. rho only has effect for K>1.
        """
        path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
        self.model.eval()

        cal_data, cal_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        cal_mu, cal_sigma, cal_pi, cal_trues = self._collect_mog_predictions(cal_loader)
        test_mu, test_sigma, test_pi, test_trues = self._collect_mog_predictions(test_loader)

        mog_decompose = getattr(self.args, 'meld_hpd_mog_decompose', True)
        calibrator = MELDHPDCalibrator(alpha=0.1, mog_decompose=mog_decompose)
        calibrator.fit(cal_trues, cal_mu, cal_sigma, cal_pi)
        intervals = calibrator.predict(test_mu, test_sigma, test_pi)  # list of [m_i,2] arrays

        test_trues_np = test_trues.squeeze().numpy()

        widths = np.array([seg[:, 1].sum() - seg[:, 0].sum() if len(seg) else 0.0 for seg in intervals])
        covered = np.array([bool(len(seg)) and bool(np.any((seg[:, 0] <= y) & (y <= seg[:, 1])))
                             for y, seg in zip(test_trues_np, intervals)])
        n_segments = np.array([len(seg) for seg in intervals])

        coverage = covered.mean()
        width = widths.mean()
        median_width = np.median(widths)
        frac_multi = np.mean(n_segments > 1)
        frac_empty = np.mean(n_segments == 0)
        avg_segments = n_segments.mean()

        # Header is "MELD-HPD Results" -- shares no contiguous substring with "MoG-HPD",
        # "STA-HPD", or "SETA-HPD", so the run_tabular_regression.sh parser patterns cannot
        # cross-match this method.
        print(f"\nMELD-HPD Results (c*={calibrator.c_:.4f}, rho*={calibrator.rho_:.4f}, "
              f"theta*={calibrator.theta_:.4f}, rho_hat_closed={calibrator.rho_hat_closed_:.4f}, "
              f"mog_decompose={calibrator.mog_decompose}, "
              f"tune width@best={calibrator.tune_width_at_best_:.4f}, t_hat={calibrator.t_hat:.4f}, "
              f"G={calibrator.G_:.6g}, avg segments={avg_segments:.3f}, "
              f"frac multi-segment={frac_multi:.4f}, frac empty={frac_empty:.4f}, "
              f"n_tune={calibrator.n_tune_}, n_calib={calibrator.n_calib_}, "
              f"n_repeats={calibrator.n_repeats_}, single_fold={calibrator.single_fold_used_}):")
        print(f"Coverage: {coverage:.4f}")
        print(f"Avg Width: {width:.4f}")
        print(f"Median Width: {median_width:.4f}")

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open("result_calibration_meld_hpd.txt", 'a') as f:
            f.write(f"{setting} (MELD-HPD)\n")
            f.write(f"Coverage: {coverage:.4f}, Width: {width:.4f}, Median Width: {median_width:.4f}, "
                    f"c: {calibrator.c_:.4f}, rho: {calibrator.rho_:.4f}, "
                    f"theta: {calibrator.theta_:.4f}, rho_hat_closed: {calibrator.rho_hat_closed_:.4f}, "
                    f"mog_decompose: {calibrator.mog_decompose}, "
                    f"t_hat: {calibrator.t_hat:.4f}, avg_segments: {avg_segments:.3f}, "
                    f"frac_multi: {frac_multi:.4f}, frac_empty: {frac_empty:.4f}, "
                    f"single_fold: {calibrator.single_fold_used_}\n\n")

        np.save(folder_path + 'intervals_meld_hpd.npy', np.array(intervals, dtype=object), allow_pickle=True)

        # Diagnostics: the (c, rho) LOG-DENSITY surface (the selection objective), a rho-marginal
        # of it, the (c*,rho*,theta*) tune width, the closed-form rho anchor, and MoG-HPD, STA-HPD
        # and SETA-HPD on the identical cal/test split -- the three methods MELD-HPD is measured
        # against.
        try:
            ls_keys = sorted(calibrator.logscore_surface_.keys())
            diag_c_edge = calibrator.c_ in (calibrator.c_grid[0], calibrator.c_grid[-1])
            diag_rho_edge = calibrator.mog_decompose and \
                calibrator.rho_ in (calibrator.rho_grid[0], calibrator.rho_grid[-1])
            diag_th_edge = calibrator.theta_ in (calibrator.theta_grid[0], calibrator.theta_grid[-1])
            diag_is_boundary = diag_c_edge or diag_rho_edge or diag_th_edge
            ls_vals = [calibrator.logscore_surface_[k] for k in ls_keys]
            diag_ls_str = ", ".join(f"({c:g},{rho:g}):{calibrator.logscore_surface_[(c, rho)]:.4f}"
                                    for c, rho in ls_keys)
            # rho-marginal of the log-density: best (max) mean log-density over c, per rho.
            diag_rho_marg = {}
            for (c, rho), v in calibrator.logscore_surface_.items():
                diag_rho_marg[rho] = max(diag_rho_marg.get(rho, -np.inf), v)
            diag_rho_marg_str = ", ".join(f"{rho:g}:{diag_rho_marg[rho]:.4f}"
                                          for rho in sorted(diag_rho_marg))
            diag_votes = sorted(calibrator.vote_counts_.items(), key=lambda kv: -kv[1])
            diag_votes_str = ", ".join(f"({c:g},{rho:g},{th:g}):{v}" for (c, rho, th), v in diag_votes)

            ref_hpd = MoG_HPD_Calibrator(alpha=0.1)
            ref_hpd.fit(cal_trues, cal_mu, cal_sigma, cal_pi)
            ref_intervals = ref_hpd.predict(test_mu, test_sigma, test_pi)
            ref_widths = np.array([seg[:, 1].sum() - seg[:, 0].sum() if len(seg) else 0.0
                                   for seg in ref_intervals])
            ref_covered = np.array([bool(len(seg)) and bool(np.any((seg[:, 0] <= y) & (y <= seg[:, 1])))
                                     for y, seg in zip(test_trues_np, ref_intervals)])
            ref_width = float(ref_widths.mean())
            ref_coverage = float(ref_covered.mean())

            ref_sta = STAHPDCalibrator(alpha=0.1, verbose=False)
            ref_sta.fit(cal_trues, cal_mu, cal_sigma, cal_pi)
            sta_intervals = ref_sta.predict(test_mu, test_sigma, test_pi)
            sta_widths = np.array([seg[:, 1].sum() - seg[:, 0].sum() if len(seg) else 0.0
                                   for seg in sta_intervals])
            sta_covered = np.array([bool(len(seg)) and bool(np.any((seg[:, 0] <= y) & (y <= seg[:, 1])))
                                     for y, seg in zip(test_trues_np, sta_intervals)])
            sta_width = float(sta_widths.mean())
            sta_coverage = float(sta_covered.mean())

            # SETA-HPD (width-selected rho) on the same split -- the method MELD-HPD's density-based
            # selection is meant to improve on.
            ref_seta = SETAHPDCalibrator(alpha=0.1, verbose=False)
            ref_seta.fit(cal_trues, cal_mu, cal_sigma, cal_pi)
            seta_intervals = ref_seta.predict(test_mu, test_sigma, test_pi)
            seta_widths = np.array([seg[:, 1].sum() - seg[:, 0].sum() if len(seg) else 0.0
                                    for seg in seta_intervals])
            seta_covered = np.array([bool(len(seg)) and bool(np.any((seg[:, 0] <= y) & (y <= seg[:, 1])))
                                      for y, seg in zip(test_trues_np, seta_intervals)])
            seta_width = float(seta_widths.mean())
            seta_coverage = float(seta_covered.mean())

            print("\n" + "="*40)
            print("--- MELD-HPD Diagnostics ---")
            print(f"[1] (c,rho) log-density surface ({len(ls_keys)} cells, mog_decompose={calibrator.mog_decompose}, "
                  f"{'SINGLE-FOLD (in-sample)' if calibrator.single_fold_used_ else f'{calibrator.n_repeats_} tune/calib splits'}): "
                  f"chosen (c*,rho*,theta*)=({calibrator.c_:.4f},{calibrator.rho_:.4f},{calibrator.theta_:.4f}) "
                  f"({'BOUNDARY' if diag_is_boundary else 'interior'} solution), rho_hat_closed={calibrator.rho_hat_closed_:.4f}, "
                  f"tune width@best={calibrator.tune_width_at_best_:.4f}, "
                  f"log-density range [{min(ls_vals):.4f}, {max(ls_vals):.4f}]")
            print(f"    vote spread (cell:n_splits): {diag_votes_str}")
            print(f"    rho-marginal (best mean log-density over c | rho): {diag_rho_marg_str}")
            print(f"    full (c,rho) log-density surface: {diag_ls_str}")
            print(f"[2] Same-split references: MoG-HPD coverage={ref_coverage:.4f} width={ref_width:.4f} "
                  f"({100.0 * (width - ref_width) / ref_width:+.1f}% vs MELD); "
                  f"STA-HPD coverage={sta_coverage:.4f} width={sta_width:.4f} "
                  f"({100.0 * (width - sta_width) / sta_width:+.1f}% vs MELD); "
                  f"SETA-HPD coverage={seta_coverage:.4f} width={seta_width:.4f} "
                  f"({100.0 * (width - seta_width) / seta_width:+.1f}% vs MELD)  | "
                  f"MELD-HPD (this run): coverage={coverage:.4f}, avg width={width:.4f}")
            print("="*40 + "\n")

            with open("result_calibration_meld_hpd.txt", 'a') as f:
                f.write(f"{setting} (MELD-HPD Diagnostics)\n")
                f.write(f"boundary={diag_is_boundary}, chosen_c={calibrator.c_:.4f}, "
                        f"chosen_rho={calibrator.rho_:.4f}, chosen_theta={calibrator.theta_:.4f}, "
                        f"rho_hat_closed={calibrator.rho_hat_closed_:.4f}, "
                        f"mog_decompose={calibrator.mog_decompose}, "
                        f"single_fold={calibrator.single_fold_used_}, "
                        f"ref_mog_hpd: coverage={ref_coverage:.4f} width={ref_width:.4f} | "
                        f"ref_sta_hpd: coverage={sta_coverage:.4f} width={sta_width:.4f} | "
                        f"ref_seta_hpd: coverage={seta_coverage:.4f} width={seta_width:.4f} | "
                        f"meld_hpd: coverage={coverage:.4f} width={width:.4f}\n")
                f.write(f"votes (c,rho,theta):n_splits: {diag_votes_str}\n")
                f.write(f"rho_marginal (rho:best_mean_logdensity): {diag_rho_marg_str}\n")
                f.write(f"logdensity_surface ((c,rho):mean_logdensity): {diag_ls_str}\n\n")
        except Exception as e:
            print(f"[MELD-HPD Diagnostics] skipped due to error: {e}")

        return coverage, width

    def calibrate_ease_hpd(self, setting):
        """
        EASE-HPD: same (c, rho, theta) surrogate family as SETA-HPD, plus a fourth exponent phi
        tilting the threshold field by the point's EPISTEMIC SHARE E(x)/(A(x)+E(x)) (deviation
        from its calibration-set mean), instead of only the total scale sigma_tot that theta
        already sees. Selected by the same width-argmin + majority-vote machinery as STA-HPD /
        SETA-HPD (NOT a proper scoring rule like MELD-HPD), so phi=0 is always on the grid and
        EASE-HPD's tuning objective can never be worse than SETA-HPD's. Runs independently of (and
        alongside) calibrate_sta_hpd / calibrate_seta_hpd / calibrate_meld_hpd -- separate
        calibrator, separate result file. rho and phi only have effect for K>1.
        """
        path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
        self.model.eval()

        cal_data, cal_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        cal_mu, cal_sigma, cal_pi, cal_trues = self._collect_mog_predictions(cal_loader)
        test_mu, test_sigma, test_pi, test_trues = self._collect_mog_predictions(test_loader)

        mog_decompose = getattr(self.args, 'ease_hpd_mog_decompose', True)
        calibrator = EASEHPDCalibrator(alpha=0.1, mog_decompose=mog_decompose)
        calibrator.fit(cal_trues, cal_mu, cal_sigma, cal_pi)
        intervals = calibrator.predict(test_mu, test_sigma, test_pi)  # list of [m_i,2] arrays

        test_trues_np = test_trues.squeeze().numpy()

        widths = np.array([seg[:, 1].sum() - seg[:, 0].sum() if len(seg) else 0.0 for seg in intervals])
        covered = np.array([bool(len(seg)) and bool(np.any((seg[:, 0] <= y) & (y <= seg[:, 1])))
                             for y, seg in zip(test_trues_np, intervals)])
        n_segments = np.array([len(seg) for seg in intervals])

        coverage = covered.mean()
        width = widths.mean()
        median_width = np.median(widths)
        frac_multi = np.mean(n_segments > 1)
        frac_empty = np.mean(n_segments == 0)
        avg_segments = n_segments.mean()

        # Header is "EASE-HPD Results" -- shares no contiguous substring with "MoG-HPD",
        # "STA-HPD", "SETA-HPD", or "MELD-HPD", so the run_tabular_regression.sh parser patterns
        # cannot cross-match this method.
        print(f"\nEASE-HPD Results (c*={calibrator.c_:.4f}, rho*={calibrator.rho_:.4f}, "
              f"theta*={calibrator.theta_:.4f}, phi*={calibrator.phi_:.4f}, "
              f"s_bar={calibrator.s_bar_:.4f}, mog_decompose={calibrator.mog_decompose}, "
              f"tune width@best={calibrator.tune_width_at_best_:.4f}, t_hat={calibrator.t_hat:.4f}, "
              f"G={calibrator.G_:.6g}, avg segments={avg_segments:.3f}, "
              f"frac multi-segment={frac_multi:.4f}, frac empty={frac_empty:.4f}, "
              f"n_tune={calibrator.n_tune_}, n_calib={calibrator.n_calib_}, "
              f"n_repeats={calibrator.n_repeats_}, single_fold={calibrator.single_fold_used_}):")
        print(f"Coverage: {coverage:.4f}")
        print(f"Avg Width: {width:.4f}")
        print(f"Median Width: {median_width:.4f}")

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open("result_calibration_ease_hpd.txt", 'a') as f:
            f.write(f"{setting} (EASE-HPD)\n")
            f.write(f"Coverage: {coverage:.4f}, Width: {width:.4f}, Median Width: {median_width:.4f}, "
                    f"c: {calibrator.c_:.4f}, rho: {calibrator.rho_:.4f}, "
                    f"theta: {calibrator.theta_:.4f}, phi: {calibrator.phi_:.4f}, "
                    f"s_bar: {calibrator.s_bar_:.4f}, mog_decompose: {calibrator.mog_decompose}, "
                    f"t_hat: {calibrator.t_hat:.4f}, avg_segments: {avg_segments:.3f}, "
                    f"frac_multi: {frac_multi:.4f}, frac_empty: {frac_empty:.4f}, "
                    f"single_fold: {calibrator.single_fold_used_}\n\n")

        np.save(folder_path + 'intervals_ease_hpd.npy', np.array(intervals, dtype=object), allow_pickle=True)

        # Diagnostics: the chosen (c,rho,theta,phi) cell, a phi-marginal of tune width (does the
        # epistemic-share knob carry signal at all?), and MoG-HPD, STA-HPD and SETA-HPD on the
        # identical cal/test split -- the three methods EASE-HPD is measured against (SETA-HPD is
        # the phi=0 slice EASE-HPD must, by construction, never lose to on the tuning objective).
        try:
            diag_c_edge = calibrator.c_ in (calibrator.c_grid[0], calibrator.c_grid[-1])
            diag_rho_edge = calibrator.mog_decompose and \
                calibrator.rho_ in (calibrator.rho_grid[0], calibrator.rho_grid[-1])
            diag_th_edge = calibrator.theta_ in (calibrator.theta_grid[0], calibrator.theta_grid[-1])
            diag_phi_edge = calibrator.phi_active_ and \
                calibrator.phi_ in (calibrator.phi_grid[0], calibrator.phi_grid[-1])
            diag_is_boundary = diag_c_edge or diag_rho_edge or diag_th_edge or diag_phi_edge
            diag_widths = list(calibrator.tuning_widths_.values())
            diag_phi_marg = {}
            for (c, rho, th, ph), w in calibrator.tuning_widths_.items():
                diag_phi_marg[ph] = min(diag_phi_marg.get(ph, np.inf), w)
            diag_phi_marg_str = ", ".join(f"{ph:g}:{diag_phi_marg[ph]:.4f}"
                                          for ph in sorted(diag_phi_marg))
            diag_votes = sorted(calibrator.vote_counts_.items(), key=lambda kv: -kv[1])[:5]
            diag_votes_str = ", ".join(f"({c:g},{rho:g},{th:g},{ph:g}):{v}"
                                       for (c, rho, th, ph), v in diag_votes)

            ref_hpd = MoG_HPD_Calibrator(alpha=0.1)
            ref_hpd.fit(cal_trues, cal_mu, cal_sigma, cal_pi)
            ref_intervals = ref_hpd.predict(test_mu, test_sigma, test_pi)
            ref_widths = np.array([seg[:, 1].sum() - seg[:, 0].sum() if len(seg) else 0.0
                                   for seg in ref_intervals])
            ref_covered = np.array([bool(len(seg)) and bool(np.any((seg[:, 0] <= y) & (y <= seg[:, 1])))
                                     for y, seg in zip(test_trues_np, ref_intervals)])
            ref_width = float(ref_widths.mean())
            ref_coverage = float(ref_covered.mean())

            ref_sta = STAHPDCalibrator(alpha=0.1, verbose=False)
            ref_sta.fit(cal_trues, cal_mu, cal_sigma, cal_pi)
            sta_intervals = ref_sta.predict(test_mu, test_sigma, test_pi)
            sta_widths = np.array([seg[:, 1].sum() - seg[:, 0].sum() if len(seg) else 0.0
                                   for seg in sta_intervals])
            sta_covered = np.array([bool(len(seg)) and bool(np.any((seg[:, 0] <= y) & (y <= seg[:, 1])))
                                     for y, seg in zip(test_trues_np, sta_intervals)])
            sta_width = float(sta_widths.mean())
            sta_coverage = float(sta_covered.mean())

            # SETA-HPD (phi=0 slice) on the same split -- the method EASE-HPD's epistemic-share
            # field is meant to improve on, and can never lose to on the tuning objective.
            ref_seta = SETAHPDCalibrator(alpha=0.1, verbose=False)
            ref_seta.fit(cal_trues, cal_mu, cal_sigma, cal_pi)
            seta_intervals = ref_seta.predict(test_mu, test_sigma, test_pi)
            seta_widths = np.array([seg[:, 1].sum() - seg[:, 0].sum() if len(seg) else 0.0
                                    for seg in seta_intervals])
            seta_covered = np.array([bool(len(seg)) and bool(np.any((seg[:, 0] <= y) & (y <= seg[:, 1])))
                                      for y, seg in zip(test_trues_np, seta_intervals)])
            seta_width = float(seta_widths.mean())
            seta_coverage = float(seta_covered.mean())

            print("\n" + "="*40)
            print("--- EASE-HPD Diagnostics ---")
            print(f"[1] (c,rho,theta,phi) surface ({len(diag_widths)} cells, mog_decompose={calibrator.mog_decompose}, "
                  f"phi_active={calibrator.phi_active_}, "
                  f"{'SINGLE-FOLD (in-sample)' if calibrator.single_fold_used_ else f'{calibrator.n_repeats_} tune/calib splits'}): "
                  f"chosen (c*,rho*,theta*,phi*)=({calibrator.c_:.4f},{calibrator.rho_:.4f},"
                  f"{calibrator.theta_:.4f},{calibrator.phi_:.4f}) "
                  f"({'BOUNDARY' if diag_is_boundary else 'interior'} solution), s_bar={calibrator.s_bar_:.4f}, "
                  f"tune width@best={calibrator.tune_width_at_best_:.4f}, "
                  f"width range [{min(diag_widths):.4f}, {max(diag_widths):.4f}]")
            print(f"    top votes (cell:n_splits): {diag_votes_str}")
            print(f"    phi-marginal (best tune width over c,rho,theta | phi): {diag_phi_marg_str}")
            print(f"[2] Same-split references: MoG-HPD coverage={ref_coverage:.4f} width={ref_width:.4f} "
                  f"({100.0 * (width - ref_width) / ref_width:+.1f}% vs EASE); "
                  f"STA-HPD coverage={sta_coverage:.4f} width={sta_width:.4f} "
                  f"({100.0 * (width - sta_width) / sta_width:+.1f}% vs EASE); "
                  f"SETA-HPD coverage={seta_coverage:.4f} width={seta_width:.4f} "
                  f"({100.0 * (width - seta_width) / seta_width:+.1f}% vs EASE)  | "
                  f"EASE-HPD (this run): coverage={coverage:.4f}, avg width={width:.4f}")
            print("="*40 + "\n")

            with open("result_calibration_ease_hpd.txt", 'a') as f:
                f.write(f"{setting} (EASE-HPD Diagnostics)\n")
                f.write(f"boundary={diag_is_boundary}, chosen_c={calibrator.c_:.4f}, "
                        f"chosen_rho={calibrator.rho_:.4f}, chosen_theta={calibrator.theta_:.4f}, "
                        f"chosen_phi={calibrator.phi_:.4f}, s_bar={calibrator.s_bar_:.4f}, "
                        f"mog_decompose={calibrator.mog_decompose}, phi_active={calibrator.phi_active_}, "
                        f"single_fold={calibrator.single_fold_used_}, "
                        f"ref_mog_hpd: coverage={ref_coverage:.4f} width={ref_width:.4f} | "
                        f"ref_sta_hpd: coverage={sta_coverage:.4f} width={sta_width:.4f} | "
                        f"ref_seta_hpd: coverage={seta_coverage:.4f} width={seta_width:.4f} | "
                        f"ease_hpd: coverage={coverage:.4f} width={width:.4f}\n")
                f.write(f"top_votes (c,rho,theta,phi):n_splits: {diag_votes_str}\n")
                f.write(f"phi_marginal (phi:best_tune_width): {diag_phi_marg_str}\n\n")
        except Exception as e:
            print(f"[EASE-HPD Diagnostics] skipped due to error: {e}")

        return coverage, width

    def calibrate_moce(self, setting):
        path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
        self.model.eval()

        cal_data, cal_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        cal_mu, cal_pi, cal_trues = self._collect_expert_predictions(cal_loader)
        test_mu, test_pi, test_trues = self._collect_expert_predictions(test_loader)

        calibrator = MoCE_Calibrator(alpha=0.1, epsilon=self.args.moce_epsilon)
        calibrator.fit(cal_mu, cal_trues, cal_pi)
        intervals = calibrator.predict(test_mu, test_pi)

        intervals = intervals.numpy()
        test_trues = test_trues.squeeze().numpy()

        coverage = np.mean((test_trues >= intervals[:, 0]) & (test_trues <= intervals[:, 1]))
        width = np.mean(intervals[:, 1] - intervals[:, 0])
        median_width = np.median(intervals[:, 1] - intervals[:, 0])

        print(f"\nMoCE Results:")
        print(f"Coverage: {coverage:.4f}")
        print(f"Avg Width: {width:.4f}")
        print(f"Median Width: {median_width:.4f}")

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open("result_calibration_moce.txt", 'a') as f:
            f.write(f"{setting} (MoCE)\n")
            f.write(f"Coverage: {coverage:.4f}, Width: {width:.4f}, Median Width: {median_width:.4f}\n\n")

        np.save(folder_path + 'intervals_moce.npy', intervals)
        return coverage, width

