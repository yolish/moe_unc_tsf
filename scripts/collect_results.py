import os
import numpy as np 
import pandas as pd 
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TS with Unc-MoE')

    # basic config
    parser.add_argument('--results_root_path', type=str, required=False, default='results',
                        help='root path of results folder')
    parser.add_argument('--out_file', type=str, required=False, default="moe_results.csv", help='output csv for results summary')
    args = parser.parse_args()

    all_results_path = args.results_root_path
    result_folders = list(os.listdir(all_results_path))
    results_dict = {"task":[], "expert":[], "num_experts": [], "prob_moe":[], 
                    "unc_gating":[], "dataset":[], "sl":[], "pl":[], "setting":[],
                    "mse":[], "mae":[], "rmse":[], "mape":[], "mspe":[]}
    for res_folder in result_folders:
        # extract hyperparams from folder name
        setting = res_folder
        parts = setting.split("_")
        results_dict["task"].append(parts[0])
        results_dict["expert"].append(parts[2])
        results_dict["num_experts"].append(int(setting.split("ne")[1].split("_")[0]))
        results_dict["prob_moe"].append(int(setting.split("pmo")[1].split("_")[0]))
        results_dict["unc_gating"].append(int(setting.split("ug")[1].split("_")[0]))
        results_dict["dataset"].append(parts[3])
        results_dict["sl"].append(int(setting.split("sl")[1].split("_")[0]))
        results_dict["pl"].append(int(setting.split("pl")[1].split("_")[0]))
        results_dict["setting"].append(setting)
        # read metrics 
        metrics_file = os.path.join(os.path.join(all_results_path, res_folder), "metrics.npy")
        mae, mse, rmse, mape, mspe = np.load(metrics_file)
        results_dict["mse"].append(mse)
        results_dict["mae"].append(mae)
        results_dict["rmse"].append(rmse)
        results_dict["mape"].append(mape)
        results_dict["mspe"].append(mspe)

    df = pd.DataFrame(data=results_dict)
    df.to_csv(args.out_file, index=False)





        