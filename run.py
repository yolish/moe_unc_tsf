import argparse
import os
import torch
import torch.backends
from exp import exp_long_term_forecasting
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_tabular_regression import Exp_Tabular_Regression
from utils.print_args import print_args
import random
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='TS with Unc-MoE')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='PatchTST',
                        help='model name, supported options: [PatchTST]')
    
    # moe and uncertainty
    parser.add_argument('--num_experts', type=int, default=1, help="value > 1 indicates MoE")
    parser.add_argument('--prob_expert', action='store_true', help='construct probabilistic experts', default=False)
    parser.add_argument('--unc_gating', action='store_true', help='use uncertainty derived gating', default=False)
    parser.add_argument('--use_quantile_loss', action='store_true', default=False, help='train a true quantile-regression head via pinball loss (mutually exclusive with prob_expert)')
    parser.add_argument('--max_grad_norm',type=float, help='value for max grad norm for prob MoE only, ignored if <=0 ', default=0)
    parser.add_argument('--load_balance_weight', type=float, default=0.05,
                        help='coefficient for the MoE load-balancing penalty (Shazeer et al. 2017 importance '
                             'loss) applied when num_experts>1; 0.05 preserves the original/current behavior.')
    parser.add_argument('--save_expert_outputs', action='store_true', help='save weights and per expert outputs', default=False)
    parser.add_argument('--save_unc', action='store_true', help='save moe uncertainties', default=False)
    parser.add_argument('--save_outputs', action='store_true', help='save predictions and ground truth', default=False)
    parser.add_argument('--save_visuals', action='store_true', help='save visualization of predictions vs. ground truth', default=False)
    parser.add_argument('--unc_head_type', type=str, default="mlp", help='uncertainty head architecture')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/long_term_forecast/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
 
    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=2, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=2, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default='avg',
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=96,
                        help='the length of segmen-wise iteration of SegRNN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False,
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')

    # Calibration
    parser.add_argument('--do_cpvs_calibration', default=False, action='store_true', help='whether to perform CPVS calibration')
    parser.add_argument('--do_cqr_calibration', default=False, action='store_true', help='Whether to perform CQR calibration')
    parser.add_argument('--do_cp_calibration', default=False, action='store_true', help='Whether to perform CP calibration')
    parser.add_argument('--do_cqr_retrain_calibration', default=False, action='store_true',
                        help='Whether to perform rolling-window retrained CQR calibration. The model is periodically '
                             're-fit on a trailing window of recent history as the test stream advances, following the '
                             'rolling re-estimation protocol of Gibbs & Candes (Adaptive Conformal Inference, NeurIPS '
                             '2021, Sec 2.2), with the retrain cadence of EnCQR (Jensen et al., IEEE TNNLS 2022). '
                             'Requires --use_quantile_loss.')
    parser.add_argument('--do_aleatoric_mog_calibration', default=False, action='store_true',
                        help='Whether to perform Aleatoric MoG calibration (CP-VS scored by '
                             'sigma_ale * exp(var_epi / var_ale)). Requires --prob_expert.')
    parser.add_argument('--do_aleatoric_mog_calibration_second_option', default=False, action='store_true',
                        help='Whether to perform the second Aleatoric MoG variant, which scales by '
                             '(sigma_epi + sigma_ale) * exp(var_epi / var_ale). Requires --prob_expert.')
    parser.add_argument('--do_aleatoric_only_calibration', default=False, action='store_true',
                        help='Whether to perform Aleatoric Only calibration (variance scaling on the '
                             'aleatoric component alone). Requires --prob_expert.')
    parser.add_argument('--do_aleatoric_scale_calibration', default=False, action='store_true',
                        help='Whether to perform Aleatoric Scale calibration: variance scaling that '
                             'calibrates the aleatoric component only and carries the epistemic '
                             'component as an uncalibrated floor, width = sqrt(q^2*var_ale + var_epi). '
                             'Requires --prob_expert.')
    parser.add_argument('--do_aci_aleatoric_scale_calibration', default=False, action='store_true',
                        help='Whether to perform ACI Aleatoric Scale calibration: the Aleatoric Scale '
                             'score/interval with the target level driven by Adaptive Conformal '
                             'Inference (Gibbs & Candes 2021), alpha_t <- alpha_t + gamma*(alpha - err_t) '
                             'per (horizon, channel) cell, so realized coverage is steered back to '
                             'nominal under drift. gamma=0 recovers Aleatoric Scale. Requires --prob_expert.')
    parser.add_argument('--aci_gamma', type=float, default=0.01,
                        help='ACI step size, shared by every --do_aci_*_calibration method. Larger '
                             'tracks distribution shift faster at the cost of noisier interval '
                             'widths. Note the delayed-update protocol feeds back errors that stay '
                             'correlated across a whole horizon, so the effective step is '
                             '~pred_len*gamma; on long horizons consider scaling this down '
                             '(0.1/pred_len keeps alpha_t off the clip on ETTh1, i.e. ~0.001 at '
                             'pred_len=96).')
    parser.add_argument('--do_aci_aleatoric_scale_g001_calibration', default=False, action='store_true',
                        help='Second ACI Aleatoric Scale variant with gamma fixed at 0.001 regardless '
                             'of --aci_gamma, run alongside --do_aci_aleatoric_scale_calibration for '
                             'side-by-side comparison. Writes to a separate result file. Requires '
                             '--prob_expert.')
    parser.add_argument('--aci_alpha', type=float, default=0.1,
                        help='Nominal miscoverage the ACI recursion targets; intervals aim for '
                             '1 - alpha coverage. Shared by every --do_aci_*_calibration method, '
                             'including ACI MoECP (which uses this, NOT --moecp_alpha).')
    parser.add_argument('--do_aci_cp_calibration', default=False, action='store_true',
                        help='ACI-adaptive standard CP: AdaptiveCP\'s absolute-residual score with the '
                             'target level driven by the Gibbs & Candes (2021) recursion per (horizon, '
                             'channel) cell. Uses --aci_gamma/--aci_alpha. gamma=0 recovers '
                             '--do_cp_calibration exactly. Ungated: works for any model.')
    parser.add_argument('--do_aci_cpvs_calibration', default=False, action='store_true',
                        help='ACI-adaptive CP-VS: AdaptiveCPVS\'s sigma-normalized residual score with '
                             'the ACI level recursion, width = q*sigma. gamma=0 recovers '
                             '--do_cpvs_calibration exactly. Requires --prob_expert for a meaningful '
                             'sigma.')
    parser.add_argument('--do_aci_aleatoric_only_calibration', default=False, action='store_true',
                        help='ACI-adaptive aleatoric-only variance scaling: score r^2/var_ale, width '
                             'sqrt(q^2*var_ale), with the ACI level recursion. gamma=0 recovers '
                             '--do_aleatoric_only_calibration exactly. Requires --prob_expert.')
    parser.add_argument('--do_aci_cqr_calibration', default=False, action='store_true',
                        help='ACI-adaptive CQR: OnlineCQRQuantile\'s signed conformity score with the '
                             'ACI level recursion. gamma=0 recovers --do_cqr_calibration exactly. '
                             'Requires --use_quantile_loss.')
    parser.add_argument('--do_aci_cqr_retrain_calibration', default=False, action='store_true',
                        help='ACI-adaptive retrained CQR: the rolling model re-fit of Gibbs & Candes '
                             'Sec 2.2 plus the level recursion of their Sec 2.1, i.e. both halves of '
                             'the paper rather than only the retraining half. Uses --retrain_* and '
                             '--aci_gamma/--aci_alpha. gamma=0 recovers '
                             '--do_cqr_retrain_calibration exactly. Requires --use_quantile_loss.')
    parser.add_argument('--do_aci_moecp_calibration', default=False, action='store_true',
                        help='ACI-adaptive MoECP: the gate-localized weighted quantile read off at a '
                             'per-(horizon, channel) level 1 - alpha_t instead of a fixed 1 - alpha. '
                             'Unlike the other ACI variants the lower clip is not windup-prone here: '
                             'alpha_t -> 0 makes the weighted CDF fall short of its target and the '
                             'interval goes unbounded, which forces err_t = 0 and pulls alpha_t back '
                             'up. Uses --aci_gamma/--aci_alpha and --moecp_temperature/'
                             '--moecp_recompute_every. Runs serially (--moecp_workers is ignored). '
                             'gamma=0 recovers --do_moecp_calibration bit-for-bit.')
    parser.add_argument('--do_adaptive_variance_calibration', default=False, action='store_true',
                        help='Whether to perform Adaptive Variance calibration: the Aleatoric Scale '
                             'family with a learned epistemic weight r, width = sqrt(q^2*var_ale + '
                             'r*var_epi). r is tuned for width on contiguous blocks of the validation '
                             'split (r=1 is Aleatoric Scale, r=0 is Aleatoric Only). Requires --prob_expert.')
    parser.add_argument('--do_adaptive_window_calibration', default=False, action='store_true',
                        help='Whether to perform Adaptive Window calibration: the Aleatoric Scale '
                             'score/interval with the sliding-window size selected per dataset by '
                             'replaying the delayed online protocol on held-out validation steps and '
                             'minimizing the interval score. Requires --prob_expert.')
    parser.add_argument('--do_cp_dvs_calibration', default=False, action='store_true',
                        help='Whether to perform CP-DVS calibration: split conformal whose scale '
                             'field is fit by quantile regression at the target level on the MoG '
                             'law-of-total-variance decomposition (within/between components). '
                             'Standard CP and CP-VS are exact points of its model family. '
                             'Requires --prob_expert.')
    parser.add_argument('--cp_dvs_alpha', type=float, default=0.1,
                        help='Miscoverage level for CP-DVS; intervals target 1 - alpha coverage.')
    parser.add_argument('--do_moecp_calibration', default=False, action='store_true',
                        help='Whether to perform MoECP calibration for time series: weighted '
                             'conformal prediction whose calibration quantile is localized by '
                             'gate-distribution similarity, w_i = exp(-tau*KL(pi~||pi_i)). Uses '
                             'the same rolling window (1000) and delayed-update protocol as CPVS. '
                             'Can return unbounded intervals by design. Requires --prob_expert.')
    parser.add_argument('--moecp_alpha', type=float, default=0.1,
                        help='Miscoverage level for time-series MoECP.')
    parser.add_argument('--moecp_temperature', type=float, default=1.0,
                        help='MoECP tau: multinomial count for the gate randomization and the '
                             'exponent on the KL divergence. Larger localizes harder, at the cost '
                             "of the test point's own weight w~_{n+1} more often exceeding alpha "
                             '(unbounded interval). tau=1 keeps that near 0 on this TSF grid while '
                             'still localizing meaningfully (widths stay well under standard_cp\'s); '
                             'the paper\'s tabular default of 100 pushed unbounded rates as high as '
                             '25%% here. tau=0 eliminates it exactly but degenerates to uniform '
                             'weights, i.e. plain split CP on the absolute residual.')
    parser.add_argument('--moecp_recompute_every', type=int, default=1,
                        help='Recompute MoECP localized weights every k-th test origin (1 = '
                             'exact). Raise to trade fidelity for speed on long horizons.')
    parser.add_argument('--moecp_workers', type=int, default=1,
                        help='Split the MoECP channel axis across this many worker processes. '
                             'The parent keeps sole ownership of the RNG and scatters the pi~ '
                             'it draws, so results are bit-identical to the serial path (1). '
                             'Only worth raising on wide datasets: MoECP is O(W log W * H * C) '
                             'per origin, so traffic (862 channels) is ~13 s/origin serially.')
    parser.add_argument('--retrain_mode', type=str, default='finetune', choices=['finetune', 'scratch'],
                        help='finetune: warm-start from current weights (default; keeps the ACI-faithful shared '
                             'window trainable). scratch: re-initialize the model each retrain - pair with a larger '
                             '--retrain_window (>=5000), which forfeits the single-shared-window property.')
    parser.add_argument('--retrain_interval', type=int, default=None,
                        help='Retrain every this many test steps. Defaults to pred_len.')
    parser.add_argument('--retrain_window', type=int, default=1000,
                        help='Size of the trailing window (in samples) used for BOTH the model re-fit and the '
                             'conformal score buffer, per ACI Sec 2.2.')
    parser.add_argument('--retrain_epochs', type=int, default=3, help='Epochs per retrain (no early stopping)')
    parser.add_argument('--retrain_lr', type=float, default=None,
                        help='Learning rate used when retraining. Defaults to --learning_rate.')


    # Overwrite existing checkpoints and force retraining
    parser.add_argument('--overwrite', action='store_true', default=False, help='Overwrite existing checkpoints and force retraining')

    # Regression specific args
    parser.add_argument('--use_reg_moecp', action='store_true', default=False, help='Use MoECP calibration for regression')
    parser.add_argument('--use_reg_moce', action='store_true', default=False, help='Use MoCE (Mixture of Conformal Experts) calibration for regression')
    parser.add_argument('--use_reg_cp_vs', action='store_true', default=False, help='Use CP_VS calibration for regression')
    parser.add_argument('--use_reg_cp_aleatoric', action='store_true', default=False, help='Use CP_VS Aleatoric calibration')
    parser.add_argument('--use_reg_cp_aleat_scale', action='store_true', default=False, help='Scale only aleatoric uncertainty by learned q')
    parser.add_argument('--use_reg_standard_cp', action='store_true', default=False, help='Use standard CP calibration')
    parser.add_argument('--use_reg_cqr', action='store_true', default=False, help='Use CQR (conformalized quantile regression) calibration for regression')
    parser.add_argument('--use_reg_adaptive_variance', action='store_true', default=False, help='Adaptive aleatoric/epistemic variance-ratio CP with width-minimizing r search')
    parser.add_argument('--use_reg_mog_hpd', action='store_true', default=False, help='Highest-density-region (HPD) conformal calibration using the closed-form Gaussian-mixture density; supports disjoint multi-interval prediction sets (MOG/MOGU only, requires --prob_expert)')
    parser.add_argument('--use_reg_sta_hpd', action='store_true', default=False, help='Shape-Threshold Adaptive HPD: MoG-HPD generalized with a shape exponent c on the per-expert sigmas and a threshold-field exponent theta on log sigma_tot, both chosen by majority vote over repeated tune/calib splits. (c=1,theta=0) is MoG-HPD and (c=1,theta=1) is CP-VS, so it cannot be worse than either on the tuning objective (MOG/MOGU only, requires --prob_expert)')
    parser.add_argument('--use_reg_seta_hpd', action='store_true', default=False, help='SETA-HPD (Shape-Epistemic-Threshold Adaptive HPD): STA-HPD additionally reshaping the between-component (epistemic) MoG variance by a tuned exponent rho via the mean-shift tilde_mu_k=yhat+sqrt(rho)(mu_k-yhat), i.e. a 3-D (c,rho,theta) grid search. rho=1 recovers STA-HPD; runs independently of it. Only affects K>1 (multi-expert) models (MOG/MOGU only, requires --prob_expert)')
    parser.add_argument('--use_reg_meld_hpd', action='store_true', default=False, help='MELD-HPD (Maximum-likelihood Epistemic-aLeatoric Decomposition HPD): same (c,rho,theta) surrogate family as SETA-HPD, but the shape exponents (c,rho) are selected by MAXIMIZING mean calibration log predictive density (a proper scoring rule -> best density match -> narrowest valid HPD) instead of by argmin tune-fold width, while theta is still chosen by width. Runs independently of STA-HPD/SETA-HPD. Only affects K>1 (multi-expert) models (MOG/MOGU only, requires --prob_expert)')
    parser.add_argument('--use_reg_ease_hpd', action='store_true', default=False, help='EASE-HPD (Epistemic-Aleatoric Share-adaptive threshold HPD): same (c,rho,theta) surrogate family as SETA-HPD, plus a fourth exponent phi tilting the threshold field by the point epistemic SHARE E(x)/(A(x)+E(x)) (deviation from its calibration-set mean) instead of only the total scale theta already sees. Selected by the same width-argmin + majority-vote machinery as STA-HPD/SETA-HPD (phi=0 always on the grid), so its tuning objective can never be worse than SETA-HPD. Runs independently of STA-HPD/SETA-HPD/MELD-HPD. Only affects K>1 (multi-expert) models (MOG/MOGU only, requires --prob_expert)')
    parser.add_argument('--tau', type=int, default=100)
    parser.add_argument('--moce_epsilon', type=float, default=1e-3,
                        help='gating-regularization smoothing constant for MoCE calibration (Eq. 2)')
    parser.add_argument('--n_samples', type=int, default=None,
                        help='override per-split sample count for Dataset_Synthetic (tabular_regression only)')

    args = parser.parse_args()

    assert args.task_name in ['long_term_forecast', 'tabular_regression'], "Task not supported"
    args.moe = (args.num_experts > 1) or (args.prob_expert) 

    if args.unc_gating and args.num_experts == 1:
        print(">> SKIPPING: Redundant experiment (Single Expert + Unc Gating).")
        exit()
    if args.unc_gating:
        assert(args.prob_expert), "uncertainty based gating required probabilstic experts"

    if getattr(args, 'use_quantile_loss', False) and args.prob_expert:
        print(">> SKIPPING: Incompatible experiment (Quantile Loss + Probabilistic Experts).")
        exit()


    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    fix_seed = args.seed 
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'tabular_regression':
        Exp = Exp_Tabular_Regression
    # --------------------------
    else:
        Exp = Exp_Long_Term_Forecast # fallback

    dataset_name = args.data_path.replace(".csv","").replace("_","-")
    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_ne{}_pe{}_ug{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}_seed{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                dataset_name,
                int(args.num_experts),
                int(args.prob_expert),
                int(args.unc_gating),
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.expand,
                args.d_conv,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii,
                args.seed)
            
            if os.path.isdir("results/{}".format(setting)) and not args.overwrite:
                print("Already run this experiment! skip training and testing for: {}".format(setting))
            else:

                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.train(setting)

                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)
                print('>>>>>>>analyze_and_save_weights : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))

                print('>>>>>>>calibrating : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                if isinstance(exp, Exp_Long_Term_Forecast):
                    if args.moe and args.prob_expert and args.do_cpvs_calibration:
                        exp.calibrate_cpvs(setting)
                    if args.use_quantile_loss and args.do_cqr_calibration:
                        if args.prob_expert:
                            print(f"Skipping CQR for setting {setting}: Pinball loss is incompatible with prob experts")
                        else:
                            print(f"Running CQR calibration for {setting}...")
                            exp.calibrate_cqr(setting)
                    if args.do_cp_calibration:
                        print(f"Running CP calibration for {setting}...")
                        exp.calibrate_cp(setting)
                    if args.use_quantile_loss and args.do_cqr_retrain_calibration:
                        if args.prob_expert:
                            print(f"Skipping Retrained CQR for setting {setting}: Pinball loss is incompatible with prob experts")
                        else:
                            print(f"Running Retrained CQR calibration for {setting}...")
                            exp.calibrate_cqr_retrain(setting)
                    if args.do_aleatoric_mog_calibration:
                        if not (args.moe and args.prob_expert):
                            print(f"Skipping Aleatoric MoG calibration for {setting}: requires --prob_expert "
                                  f"(no separated aleatoric/epistemic uncertainty otherwise).")
                        else:
                            print(f"Running Aleatoric MoG calibration for {setting}...")
                            exp.calibrate_aleatoric_mog(setting)
                    if args.do_aleatoric_mog_calibration_second_option:
                        if not (args.moe and args.prob_expert):
                            print(f"Skipping Aleatoric MoG (second option) calibration for {setting}: requires "
                                  f"--prob_expert (no separated aleatoric/epistemic uncertainty otherwise).")
                        else:
                            print(f"Running Aleatoric MoG (second option) calibration for {setting}...")
                            exp.calibrate_aleatoric_mog_second_option(setting)
                    if args.do_aleatoric_only_calibration:
                        if not (args.moe and args.prob_expert):
                            print(f"Skipping Aleatoric Only calibration for {setting}: requires --prob_expert "
                                  f"(no separated aleatoric/epistemic uncertainty otherwise).")
                        else:
                            print(f"Running Aleatoric Only calibration for {setting}...")
                            exp.calibrate_aleatoric_only(setting)
                    if args.do_aleatoric_scale_calibration:
                        if not (args.moe and args.prob_expert):
                            print(f"Skipping Aleatoric Scale calibration for {setting}: requires --prob_expert "
                                  f"(no separated aleatoric/epistemic uncertainty otherwise).")
                        else:
                            print(f"Running Aleatoric Scale calibration for {setting}...")
                            exp.calibrate_aleatoric_scale(setting)
                    if args.do_aci_aleatoric_scale_calibration:
                        if not (args.moe and args.prob_expert):
                            print(f"Skipping ACI Aleatoric Scale calibration for {setting}: requires --prob_expert "
                                  f"(no separated aleatoric/epistemic uncertainty otherwise).")
                        else:
                            print(f"Running ACI Aleatoric Scale calibration for {setting}...")
                            exp.calibrate_aci_aleatoric_scale(setting)
                    if args.do_aci_aleatoric_scale_g001_calibration:
                        if not (args.moe and args.prob_expert):
                            print(f"Skipping ACI Aleatoric Scale (g=0.001) calibration for {setting}: requires "
                                  f"--prob_expert (no separated aleatoric/epistemic uncertainty otherwise).")
                        else:
                            print(f"Running ACI Aleatoric Scale (g=0.001) calibration for {setting}...")
                            exp.calibrate_aci_aleatoric_scale_g001(setting)
                    if args.do_adaptive_variance_calibration:
                        if not (args.moe and args.prob_expert):
                            print(f"Skipping Adaptive Variance calibration for {setting}: requires --prob_expert "
                                  f"(no separated aleatoric/epistemic uncertainty otherwise).")
                        else:
                            print(f"Running Adaptive Variance calibration for {setting}...")
                            exp.calibrate_adaptive_variance(setting)
                    if args.do_adaptive_window_calibration:
                        if not (args.moe and args.prob_expert):
                            print(f"Skipping Adaptive Window calibration for {setting}: requires --prob_expert "
                                  f"(no separated aleatoric/epistemic uncertainty otherwise).")
                        else:
                            print(f"Running Adaptive Window calibration for {setting}...")
                            exp.calibrate_adaptive_window(setting)
                    if args.do_cp_dvs_calibration:
                        if not (args.moe and args.prob_expert):
                            print(f"Skipping CP-DVS calibration for {setting}: requires --prob_expert "
                                  f"(no MoG variance decomposition otherwise).")
                        else:
                            print(f"Running CP-DVS calibration for {setting}...")
                            exp.calibrate_cp_dvs(setting)
                    if args.do_moecp_calibration:
                        if not args.moe or (args.unc_gating and not args.prob_expert):
                            print(f"Skipping MoECP calibration for {setting}: requires a softmax "
                                  f"gating distribution (num_experts > 1, and --prob_expert if "
                                  f"--unc_gating is set).")
                        else:
                            print(f"Running MoECP calibration for {setting}...")
                            exp.calibrate_moecp(setting)

                    # ACI-adaptive variants. Gating conditions are copied verbatim from each
                    # base method's block above so the two stay in lockstep.
                    if args.do_aci_cp_calibration:
                        print(f"Running ACI CP calibration for {setting}...")
                        exp.calibrate_aci_cp(setting)
                    if args.moe and args.prob_expert and args.do_aci_cpvs_calibration:
                        print(f"Running ACI CP-VS calibration for {setting}...")
                        exp.calibrate_aci_cpvs(setting)
                    if args.do_aci_aleatoric_only_calibration:
                        if not (args.moe and args.prob_expert):
                            print(f"Skipping ACI Aleatoric Only calibration for {setting}: requires "
                                  f"--prob_expert (no MoG variance decomposition otherwise).")
                        else:
                            print(f"Running ACI Aleatoric Only calibration for {setting}...")
                            exp.calibrate_aci_aleatoric_only(setting)
                    if args.use_quantile_loss and args.do_aci_cqr_calibration:
                        if args.prob_expert:
                            print(f"Skipping ACI CQR for setting {setting}: Pinball loss is incompatible with prob experts")
                        else:
                            print(f"Running ACI CQR calibration for {setting}...")
                            exp.calibrate_aci_cqr(setting)
                    if args.use_quantile_loss and args.do_aci_cqr_retrain_calibration:
                        if args.prob_expert:
                            print(f"Skipping ACI Retrained CQR for setting {setting}: Pinball loss is incompatible with prob experts")
                        else:
                            print(f"Running ACI Retrained CQR calibration for {setting}...")
                            exp.calibrate_aci_cqr_retrain(setting)
                    if args.do_aci_moecp_calibration:
                        if not args.moe or (args.unc_gating and not args.prob_expert):
                            print(f"Skipping ACI MoECP calibration for {setting}: requires a softmax "
                                  f"gating distribution (num_experts > 1, and --prob_expert if "
                                  f"--unc_gating is set).")
                        else:
                            print(f"Running ACI MoECP calibration for {setting}...")
                            exp.calibrate_aci_moecp(setting)

                elif isinstance(exp, Exp_Tabular_Regression):
                    if args.use_reg_moecp:
                        print(f"Running MoECP calibration for regression for setting {setting}...")
                        exp.calibrate_moecp(setting)
                    if args.use_reg_moce:
                        print(f"Running MoCE calibration for regression for setting {setting}...")
                        exp.calibrate_moce(setting)
                    if args.use_reg_cp_vs:
                        print(f"Running CP_VS calibration for regression for setting {setting}...")
                        exp.calibrate_cpvs(setting)
                    if args.use_reg_cp_aleatoric:
                        print(f"Running CP_VS aleatoric calibration for regression for setting {setting}...")
                        exp.calibrate_cpvs_aleatoric(setting)
                    if args.use_reg_cp_aleat_scale:
                        print(f"Running CP Aleatoric Scaling calibration for setting {setting}...")
                        exp.calibrate_cp_aleatoric_scale(setting)
                    if args.use_reg_standard_cp:
                        print(f"Running Standard CP calibration for setting {setting}...")
                        exp.calibrate_standard_cp(setting)
                    if args.use_reg_cqr:
                        print(f"Running CQR calibration for setting {setting}...")
                        exp.calibrate_cqr(setting)
                    if args.use_reg_adaptive_variance:
                        print(f"Running Adaptive Variance-Ratio calibration for setting {setting}...")
                        exp.calibrate_adaptive_variance(setting)
                    if args.use_reg_mog_hpd:
                        if not args.prob_expert:
                            print(f"Skipping MoG-HPD calibration for setting {setting}: requires --prob_expert "
                                  f"(MOG/MOGU); ClassicMoE has no per-expert variance.")
                        else:
                            print(f"Running MoG-HPD calibration for setting {setting}...")
                            exp.calibrate_mog_hpd(setting)
                    if args.use_reg_sta_hpd:
                        if not args.prob_expert:
                            print(f"Skipping STA-HPD calibration for setting {setting}: requires "
                                  f"--prob_expert (MOG/MOGU); ClassicMoE has no per-expert variance.")
                        else:
                            print(f"Running STA-HPD calibration for setting {setting}...")
                            exp.calibrate_sta_hpd(setting)
                    if args.use_reg_seta_hpd:
                        if not args.prob_expert:
                            print(f"Skipping SETA-HPD calibration for setting {setting}: requires "
                                  f"--prob_expert (MOG/MOGU); ClassicMoE has no per-expert variance.")
                        else:
                            print(f"Running SETA-HPD calibration for setting {setting}...")
                            exp.calibrate_seta_hpd(setting)
                    if args.use_reg_meld_hpd:
                        if not args.prob_expert:
                            print(f"Skipping MELD-HPD calibration for setting {setting}: requires "
                                  f"--prob_expert (MOG/MOGU); ClassicMoE has no per-expert variance.")
                        else:
                            print(f"Running MELD-HPD calibration for setting {setting}...")
                            exp.calibrate_meld_hpd(setting)
                    if args.use_reg_ease_hpd:
                        if not args.prob_expert:
                            print(f"Skipping EASE-HPD calibration for setting {setting}: requires "
                                  f"--prob_expert (MOG/MOGU); ClassicMoE has no per-expert variance.")
                        else:
                            print(f"Running EASE-HPD calibration for setting {setting}...")
                            exp.calibrate_ease_hpd(setting)

            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
    else:
        exp = Exp(args)  # set experiments
        ii = 0
        setting = '{}_{}_{}_{}_ne{}_pe{}_ug{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}_seed{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            dataset_name,
            int(args.num_experts),
            int(args.prob_expert),
            int(args.unc_gating),
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.expand,
            args.d_conv,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii,
            args.seed)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        print('>>>>>>>calibrating : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        if isinstance(exp, Exp_Long_Term_Forecast):
            if args.moe and args.prob_expert and args.do_cpvs_calibration:
                exp.calibrate_cpvs(setting)
            if args.use_quantile_loss and args.do_cqr_calibration:
                if args.prob_expert:
                    print(f"Skipping CQR for setting {setting}: Pinball loss is incompatible with prob experts")
                else:
                    print(f"Running CQR calibration for {setting}...")
                    exp.calibrate_cqr(setting)
            if args.do_cp_calibration:
                print(f"Running CP calibration for {setting}...")
                exp.calibrate_cp(setting)
            if args.use_quantile_loss and args.do_cqr_retrain_calibration:
                if args.prob_expert:
                    print(f"Skipping Retrained CQR for setting {setting}: Pinball loss is incompatible with prob experts")
                else:
                    print(f"Running Retrained CQR calibration for {setting}...")
                    exp.calibrate_cqr_retrain(setting)
            if args.do_aleatoric_mog_calibration:
                if not (args.moe and args.prob_expert):
                    print(f"Skipping Aleatoric MoG calibration for {setting}: requires --prob_expert "
                          f"(no separated aleatoric/epistemic uncertainty otherwise).")
                else:
                    print(f"Running Aleatoric MoG calibration for {setting}...")
                    exp.calibrate_aleatoric_mog(setting)
            if args.do_aleatoric_mog_calibration_second_option:
                if not (args.moe and args.prob_expert):
                    print(f"Skipping Aleatoric MoG (second option) calibration for {setting}: requires "
                          f"--prob_expert (no separated aleatoric/epistemic uncertainty otherwise).")
                else:
                    print(f"Running Aleatoric MoG (second option) calibration for {setting}...")
                    exp.calibrate_aleatoric_mog_second_option(setting)
            if args.do_aleatoric_only_calibration:
                if not (args.moe and args.prob_expert):
                    print(f"Skipping Aleatoric Only calibration for {setting}: requires --prob_expert "
                          f"(no separated aleatoric/epistemic uncertainty otherwise).")
                else:
                    print(f"Running Aleatoric Only calibration for {setting}...")
                    exp.calibrate_aleatoric_only(setting)
            if args.do_aleatoric_scale_calibration:
                if not (args.moe and args.prob_expert):
                    print(f"Skipping Aleatoric Scale calibration for {setting}: requires --prob_expert "
                          f"(no separated aleatoric/epistemic uncertainty otherwise).")
                else:
                    print(f"Running Aleatoric Scale calibration for {setting}...")
                    exp.calibrate_aleatoric_scale(setting)
            if args.do_aci_aleatoric_scale_calibration:
                if not (args.moe and args.prob_expert):
                    print(f"Skipping ACI Aleatoric Scale calibration for {setting}: requires --prob_expert "
                          f"(no separated aleatoric/epistemic uncertainty otherwise).")
                else:
                    print(f"Running ACI Aleatoric Scale calibration for {setting}...")
                    exp.calibrate_aci_aleatoric_scale(setting)
            if args.do_aci_aleatoric_scale_g001_calibration:
                if not (args.moe and args.prob_expert):
                    print(f"Skipping ACI Aleatoric Scale (g=0.001) calibration for {setting}: requires "
                          f"--prob_expert (no separated aleatoric/epistemic uncertainty otherwise).")
                else:
                    print(f"Running ACI Aleatoric Scale (g=0.001) calibration for {setting}...")
                    exp.calibrate_aci_aleatoric_scale_g001(setting)
            if args.do_adaptive_variance_calibration:
                if not (args.moe and args.prob_expert):
                    print(f"Skipping Adaptive Variance calibration for {setting}: requires --prob_expert "
                          f"(no separated aleatoric/epistemic uncertainty otherwise).")
                else:
                    print(f"Running Adaptive Variance calibration for {setting}...")
                    exp.calibrate_adaptive_variance(setting)
            if args.do_adaptive_window_calibration:
                if not (args.moe and args.prob_expert):
                    print(f"Skipping Adaptive Window calibration for {setting}: requires --prob_expert "
                          f"(no separated aleatoric/epistemic uncertainty otherwise).")
                else:
                    print(f"Running Adaptive Window calibration for {setting}...")
                    exp.calibrate_adaptive_window(setting)
            if args.do_cp_dvs_calibration:
                if not (args.moe and args.prob_expert):
                    print(f"Skipping CP-DVS calibration for {setting}: requires --prob_expert "
                          f"(no MoG variance decomposition otherwise).")
                else:
                    print(f"Running CP-DVS calibration for {setting}...")
                    exp.calibrate_cp_dvs(setting)
            if args.do_moecp_calibration:
                if not args.moe or (args.unc_gating and not args.prob_expert):
                    print(f"Skipping MoECP calibration for {setting}: requires a softmax gating "
                          f"distribution (num_experts > 1, and --prob_expert if --unc_gating is set).")
                else:
                    print(f"Running MoECP calibration for {setting}...")
                    exp.calibrate_moecp(setting)

            # ACI-adaptive variants. Gating conditions are copied verbatim from each base
            # method's block above so the two stay in lockstep.
            if args.do_aci_cp_calibration:
                print(f"Running ACI CP calibration for {setting}...")
                exp.calibrate_aci_cp(setting)
            if args.moe and args.prob_expert and args.do_aci_cpvs_calibration:
                print(f"Running ACI CP-VS calibration for {setting}...")
                exp.calibrate_aci_cpvs(setting)
            if args.do_aci_aleatoric_only_calibration:
                if not (args.moe and args.prob_expert):
                    print(f"Skipping ACI Aleatoric Only calibration for {setting}: requires "
                          f"--prob_expert (no MoG variance decomposition otherwise).")
                else:
                    print(f"Running ACI Aleatoric Only calibration for {setting}...")
                    exp.calibrate_aci_aleatoric_only(setting)
            if args.use_quantile_loss and args.do_aci_cqr_calibration:
                if args.prob_expert:
                    print(f"Skipping ACI CQR for setting {setting}: Pinball loss is incompatible with prob experts")
                else:
                    print(f"Running ACI CQR calibration for {setting}...")
                    exp.calibrate_aci_cqr(setting)
            if args.use_quantile_loss and args.do_aci_cqr_retrain_calibration:
                if args.prob_expert:
                    print(f"Skipping ACI Retrained CQR for setting {setting}: Pinball loss is incompatible with prob experts")
                else:
                    print(f"Running ACI Retrained CQR calibration for {setting}...")
                    exp.calibrate_aci_cqr_retrain(setting)
            if args.do_aci_moecp_calibration:
                if not args.moe or (args.unc_gating and not args.prob_expert):
                    print(f"Skipping ACI MoECP calibration for {setting}: requires a softmax gating "
                          f"distribution (num_experts > 1, and --prob_expert if --unc_gating is set).")
                else:
                    print(f"Running ACI MoECP calibration for {setting}...")
                    exp.calibrate_aci_moecp(setting)

        elif isinstance(exp, Exp_Tabular_Regression):
            if args.use_reg_moecp:
                print(f"Running MoECP calibration for regression for setting {setting}...")
                exp.calibrate_moecp(setting)
            if args.use_reg_moce:
                print(f"Running MoCE calibration for regression for setting {setting}...")
                exp.calibrate_moce(setting)
            if args.use_reg_cp_vs:
                print(f"Running CP_VS calibration for regression for setting {setting}...")
                exp.calibrate_cpvs(setting)
            if args.use_reg_cp_aleatoric:
                print(f"Running CP_VS aleatoric calibration for regression for setting {setting}...")
                exp.calibrate_cpvs_aleatoric(setting)
            if args.use_reg_cp_aleat_scale:
                print(f"Running CP Aleatoric Scaling calibration for setting {setting}...")
                exp.calibrate_cp_aleatoric_scale(setting)
            if args.use_reg_standard_cp:
                print(f"Running Standard CP calibration for setting {setting}...")
                exp.calibrate_standard_cp(setting)
            if args.use_reg_cqr:
                print(f"Running CQR calibration for setting {setting}...")
                exp.calibrate_cqr(setting)
            if args.use_reg_adaptive_variance:
                print(f"Running Adaptive Variance-Ratio calibration for setting {setting}...")
                exp.calibrate_adaptive_variance(setting)
            if args.use_reg_mog_hpd:
                if not args.prob_expert:
                    print(f"Skipping MoG-HPD calibration for setting {setting}: requires --prob_expert "
                          f"(MOG/MOGU); ClassicMoE has no per-expert variance.")
                else:
                    print(f"Running MoG-HPD calibration for setting {setting}...")
                    exp.calibrate_mog_hpd(setting)
            if args.use_reg_sta_hpd:
                if not args.prob_expert:
                    print(f"Skipping STA-HPD calibration for setting {setting}: requires "
                          f"--prob_expert (MOG/MOGU); ClassicMoE has no per-expert variance.")
                else:
                    print(f"Running STA-HPD calibration for setting {setting}...")
                    exp.calibrate_sta_hpd(setting)
            if args.use_reg_seta_hpd:
                if not args.prob_expert:
                    print(f"Skipping SETA-HPD calibration for setting {setting}: requires "
                          f"--prob_expert (MOG/MOGU); ClassicMoE has no per-expert variance.")
                else:
                    print(f"Running SETA-HPD calibration for setting {setting}...")
                    exp.calibrate_seta_hpd(setting)
            if args.use_reg_meld_hpd:
                if not args.prob_expert:
                    print(f"Skipping MELD-HPD calibration for setting {setting}: requires "
                          f"--prob_expert (MOG/MOGU); ClassicMoE has no per-expert variance.")
                else:
                    print(f"Running MELD-HPD calibration for setting {setting}...")
                    exp.calibrate_meld_hpd(setting)
            if args.use_reg_ease_hpd:
                if not args.prob_expert:
                    print(f"Skipping EASE-HPD calibration for setting {setting}: requires "
                          f"--prob_expert (MOG/MOGU); ClassicMoE has no per-expert variance.")
                else:
                    print(f"Running EASE-HPD calibration for setting {setting}...")
                    exp.calibrate_ease_hpd(setting)

        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()
