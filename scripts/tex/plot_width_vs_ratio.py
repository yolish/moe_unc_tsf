import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_for_setting(setting, horizon_step):
    # נתיב לתיקיית התוצאות (בהנחה שאתה מריץ מהתיקייה הראשית)
    folder_path = os.path.join('./results', setting, '')
    
    print(f"Loading data from: {folder_path}")
    print(f"Analyzing specifically for horizon step: {horizon_step} (to avoid overlapping windows)")
    
    try:
        widths = np.load(folder_path + 'step_widths.npy')
        ale_vars = np.load(folder_path + 'step_ale_vars.npy')
        epi_vars = np.load(folder_path + 'step_epi_vars.npy')
    except FileNotFoundError:
        print(f"Error: Could not find .npy files in {folder_path}.")
        print("Make sure you ran the calibration and saved step_widths.npy, step_ale_vars.npy, step_epi_vars.npy")
        return

    # חישוב הממוצעים לצעד חיזוי ספציפי בלבד
    if widths.ndim > 1:
        # לוקחים את כל הדוגמאות, רק את הצעד הספציפי, ועושים ממוצע רק על ממד הפיצ'רים
        widths_mean = np.mean(widths[:, horizon_step, :], axis=1)
        ale_mean = np.mean(ale_vars[:, horizon_step, :], axis=1)
        epi_mean = np.mean(epi_vars[:, horizon_step, :], axis=1)
    else:
        widths_mean = widths
        ale_mean = ale_vars
        epi_mean = epi_vars

    # חישוב יחס שונויות
    eps = 1e-8
    variance_ratio = ale_mean / (epi_mean + eps)

    # יצירת הגרפים
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # גרף 1: Scatter
    ax1.scatter(variance_ratio, widths_mean, alpha=0.4, color='royalblue', edgecolors='none')
    ax1.set_xscale('log')
    ax1.set_title(f'Scatter: Width vs Ratio (Horizon Step: {horizon_step})')
    ax1.set_xlabel('Variance Ratio (Aleatoric / Epistemic)')
    ax1.set_ylabel('Interval Width')
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    # גרף 2: Bins (ממוצע לפי סלים)
    mask = variance_ratio > 0
    if np.any(mask):
        min_val = np.min(variance_ratio[mask])
        max_val = np.max(variance_ratio)
        
        if min_val < max_val:
            bins = np.logspace(np.log10(min_val), np.log10(max_val), num=15)
            bin_centers = []
            bin_avg_widths = []
            
            for i in range(len(bins)-1):
                b_mask = (variance_ratio >= bins[i]) & (variance_ratio < bins[i+1])
                if np.any(b_mask):
                    bin_centers.append(np.sqrt(bins[i] * bins[i+1]))
                    bin_avg_widths.append(np.mean(widths_mean[b_mask]))
            
            if len(bin_centers) > 0:
                ax2.plot(bin_centers, bin_avg_widths, marker='o', linestyle='-', color='darkorange', linewidth=2)
                ax2.set_xscale('log')
                ax2.set_title(f'Binned Avg: Width vs Ratio (Horizon: {horizon_step})')
                ax2.set_xlabel('Variance Ratio (Aleatoric / Epistemic)')
                ax2.set_ylabel('Average Interval Width')
                ax2.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    save_path = folder_path + f'width_vs_ratio_step_{horizon_step}.png'
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot saved successfully to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot Width vs Variance Ratio for a specific setting')
    parser.add_argument('--setting', type=str, required=True, help='The folder name of the setting in ./results/')
    parser.add_argument('--horizon_step', type=int, default=0, help='The specific prediction step to analyze (e.g., 0 for 1st step)')
    
    args = parser.parse_args()
    
    plot_for_setting(args.setting, args.horizon_step)