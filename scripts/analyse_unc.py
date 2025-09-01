import os
import numpy as np 
import pandas as pd 
import argparse
import scipy


def compute_corr(error, unc, results, thr, corr_func, description):
    R, pv = corr_func(error,unc)
    if pv > thr:
        print("non-significant results for {}".format(description))
    else:
        results.append(R)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Uncertainty Analysis')

    # basic config
    parser.add_argument('--results_root_path', type=str, required=False, default='results_with_unc',
                        help='root path of results folder')
    args = parser.parse_args()
    all_results_path = args.results_root_path
    result_folders = list(os.listdir(all_results_path))
    for res_folder in result_folders:
        print(res_folder)
        epi_unc_file = os.path.join(os.path.join(all_results_path, res_folder), "epi_unc.npy")
        ale_unc_file = os.path.join(os.path.join(all_results_path, res_folder), "ale_unc.npy")
        pred_file = os.path.join(os.path.join(all_results_path, res_folder), "pred.npy")
        true_file =  os.path.join(os.path.join(all_results_path, res_folder), "true.npy")
        try:
            epi_unc = np.load(epi_unc_file)
            ale_unc = np.load(ale_unc_file)
            unc = epi_unc + ale_unc
            true = np.load(true_file)
            pred =  np.load(pred_file)
            mse = (true-pred)**2
            mae = np.abs(true-pred)

            pearson = {"total":[], "epi":[], "ale":[]}
            spearman = {"total":[], "epi":[], "ale":[]}

            significance_thr = 1e-05
            for i in range(unc.shape[-1]):
                mae_var = mae[:, :, i].flatten()
                unc_var = unc[:, :, i].flatten()
                epi_unc_var = epi_unc[:, :, i].flatten()
                ale_unc_var = ale_unc[:, :, i].flatten()

                compute_corr(mae_var, unc_var, pearson["total"], significance_thr, scipy.stats.pearsonr, "pearson-total")
                compute_corr(mae_var, epi_unc_var, pearson["epi"], significance_thr, scipy.stats.pearsonr, "pearson-epi")
                compute_corr(mae_var, ale_unc_var, pearson["ale"], significance_thr, scipy.stats.pearsonr, "pearson-ale")
                
                compute_corr(mae_var, unc_var, spearman["total"], significance_thr, scipy.stats.spearmanr, "spearman-total")
                compute_corr(mae_var, epi_unc_var, spearman["epi"], significance_thr, scipy.stats.spearmanr, "spearman-epi")
                compute_corr(mae_var, ale_unc_var, spearman["ale"], significance_thr, scipy.stats.spearmanr, "spearman-ale")
            
            print("pearson: {}, {}, {}".format(np.mean(pearson["ale"]), 
                                               np.mean(pearson["epi"]),
                                               np.mean(pearson["total"])))
            print("spearman: {}, {}, {}".format(np.mean(spearman["ale"]), 
                                               np.mean(spearman["epi"]),
                                               np.mean(spearman["total"])))
           
        except FileNotFoundError:
            pass
    