import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt



if __name__ == '__main__':
    #folder = "results/long_term_forecast_test_PatchTST_ETTh1_ne3_pmo1_ug1_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtTrue_test_0"
    folder_name = "long_term_forecast_test_iTransformer_ETTm1_ne3_pmo1_ug1_ftM_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtTrue_test_0"
    folder = os.path.join("results",folder_name)
    
    i = 6
    
    epi_unc_file = os.path.join(folder, "epi_unc.npy")
    ale_unc_file = os.path.join(folder, "ale_unc.npy")
    pred_file = os.path.join(folder, "pred.npy")
    true_file =  os.path.join(folder, "true.npy")
    
    epi_unc = np.load(epi_unc_file)
    ale_unc = np.load(ale_unc_file)
    unc = epi_unc + ale_unc
    true = np.load(true_file)
    pred =  np.load(pred_file)
    mse = (true-pred)**2
    mae = np.abs(true-pred)

    for k in [0, 10, 100]:
        my_mae = mae[k, :, i]
        my_pred = pred[k, :, i]
        my_true = true[k, :, i]
        my_unc = unc[k, :, i]
        plt.figure(figsize=(8, 6)) # Optional: set figure size
        plt.plot(my_unc)
        plt.plot(my_pred)
        plt.plot(my_true)
        plt.plot(my_mae)
        plt.legend(["Unc.", "Pred.", "True", "MAE"], loc='upper right', fontsize=7)
        plt.xlabel("Time Point")
        plt.savefig("s{}v{}_{}.jpg".format(k,i,folder_name))

    