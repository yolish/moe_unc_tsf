import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd

def analyze_variance_vs_error(results_path):
    # Define file paths based on run.py saving logic
    expert_outputs_file = os.path.join(results_path, "per_expert_outputs.npy")
    expert_unc_file = os.path.join(results_path, "per_expert_unc.npy")
    true_file = os.path.join(results_path, "true.npy")

    # Check if files exist
    if not (os.path.exists(expert_outputs_file) and os.path.exists(expert_unc_file) and os.path.exists(true_file)):
        print(f"Error: Missing files in {results_path}.")
        print("Please ensure you ran the experiment with: --save_expert_outputs --save_unc --prob_expert")
        return

    print(f"Loading results from {results_path}...")
    # Load data
    # expert_outputs shape: (Samples, Num_Experts, Pred_Len, Channels)
    # expert_unc shape: (Samples, Num_Experts, Pred_Len, Channels) -> This is Variance (sigma^2)
    # true shape: (Samples, Pred_Len, Channels)
    
    expert_outputs = np.load(expert_outputs_file)
    expert_vars = np.load(expert_unc_file) 
    true = np.load(true_file)

    # Expand true to (Samples, 1, Pred_Len, Channels) for broadcasting against experts
    true_expanded = np.expand_dims(true, 1)
    
    # Calculate Squared Prediction Error (SE) for each expert
    squared_errors = (expert_outputs - true_expanded) ** 2

    num_experts = expert_outputs.shape[1]
    print(f"Analyzing {num_experts} experts...")

    # Setup Plot
    fig, axes = plt.subplots(num_experts, 2, figsize=(15, 5 * num_experts))
    if num_experts == 1: axes = np.expand_dims(axes, 0)

    for i in range(num_experts):
        # Flatten data for expert i to treat all time steps/features as independent samples
        flat_vars = expert_vars[:, i, :, :].flatten()
        flat_errors = squared_errors[:, i, :, :].flatten()
        
        # Calculate correlations
        pearson_corr, p_val = scipy.stats.pearsonr(flat_vars, flat_errors)
        spearman_corr, s_val = scipy.stats.spearmanr(flat_vars, flat_errors)
        
        # 1. Scatter Plot (Log-Log scale often works best for Variance/Error)
        ax_scatter = axes[i, 0]
        # Downsample for plotting if too many points
        indices = np.random.choice(len(flat_vars), size=min(10000, len(flat_vars)), replace=False)
        ax_scatter.scatter(flat_vars[indices], flat_errors[indices], alpha=0.2, s=10, c='blue', edgecolors='none')
        
        ax_scatter.set_title(f'Expert {i+1}: Raw Correlation\nPearson: {pearson_corr:.3f}, Spearman: {spearman_corr:.3f}')
        ax_scatter.set_xlabel('Predicted Variance ($\sigma^2$)')
        ax_scatter.set_ylabel('Squared Error (SE)')
        ax_scatter.set_xscale('log')
        ax_scatter.set_yscale('log')
        ax_scatter.grid(True, which="both", ls="-", alpha=0.3)

        # 2. Binned Plot (To show the trend clearly)
        ax_bin = axes[i, 1]
        df = pd.DataFrame({'var': flat_vars, 'err': flat_errors})
        
        # Create bins based on quantiles of variance (e.g., 20 bins)
        # This groups points with similar uncertainty together
        try:
            df['bin'] = pd.qcut(df['var'], q=20, duplicates='drop')
            grouped = df.groupby('bin', observed=False).mean()
            
            x_vals = grouped['var']
            y_vals = grouped['err']
            
            ax_bin.plot(x_vals, y_vals, marker='o', linestyle='-', linewidth=2, color='red')
            ax_bin.set_title(f'Expert {i+1}: Binned Trend (Variance vs Error)')
            ax_bin.set_xlabel('Mean Variance (Binned)')
            ax_bin.set_ylabel('Mean Squared Error (Binned)')
            ax_bin.grid(True)
        except Exception as e:
            ax_bin.text(0.5, 0.5, f"Could not bin data: {str(e)}", ha='center')

    plt.tight_layout()
    save_file = os.path.join(results_path, 'expert_variance_vs_error.png')
    plt.savefig(save_file)
    print(f"Visualization saved to: {save_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Expert Variance vs Prediction Error')
    parser.add_argument('--results_path', type=str, required=True, 
                        help='Path to the specific results folder (e.g., results/long_term_forecast_...)')
    args = parser.parse_args()
    
    analyze_variance_vs_error(args.results_path)