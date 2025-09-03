import os
import numpy as np 
import pandas as pd 
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Weight Analysis')

    # basic config
    parser.add_argument('--results_root_path', type=str, required=False, default='results',
                        help='root path of results folder')
    args = parser.parse_args()
    all_results_path = args.results_root_path
    result_folders = ["long_term_forecast_test_PatchTST_ETTh2_ne3_pmo1_ug1_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtTrue_test_0"]#list(os.listdir(all_results_path))
    for res_folder in result_folders:
        print(res_folder)
        weights_file = os.path.join(os.path.join(all_results_path, res_folder), "weights.npy")
        true_file =  os.path.join(os.path.join(all_results_path, res_folder), "true.npy")
        per_expert_output_file = os.path.join(os.path.join(all_results_path, res_folder), "per_expert_outputs.npy")
        weights = np.load(weights_file)
        true = np.load(true_file)
        per_expert_output = np.load(per_expert_output_file)
        mse_per_expert = (per_expert_output - np.expand_dims(true,1))**2
        mse_per_expert_norm = mse_per_expert/np.expand_dims(mse_per_expert.sum(axis=1), 1)
        weights[0, :, :, 0].transpose(1,0)
        mse_per_expert_norm[0, :, :, 0].transpose(1,0)

        







        