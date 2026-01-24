import os
import numpy as np

def analyze_and_save_weights(setting, args):
    if not getattr(args, 'save_expert_outputs', False):
        return

    folder_path = './results/' + setting + '/'
    unc_path = os.path.join(folder_path, 'per_expert_unc.npy')

    if not os.path.exists(unc_path):
        return

    try:
        print(f"\n>> [Analysis] Processing Expert Weights for: {setting}")
        
        unc_data = np.load(unc_path)
        
        epsilon = 1e-8  
        inv_variance = 1.0 / (unc_data + epsilon)
        
        sum_inv_variance = np.sum(inv_variance, axis=1, keepdims=True)
        
        weights = inv_variance / sum_inv_variance
        
        avg_weights = np.mean(weights, axis=(0, 2, 3))
        
        txt_path = os.path.join(folder_path, 'expert_weights_summary.txt')
        with open(txt_path, 'w') as f:
            f.write(f"MoGU Expert Weights Analysis\n")
            f.write(f"============================\n")
            f.write(f"Setting: {setting}\n")
            f.write(f"Total Experts: {len(avg_weights)}\n\n")
            f.write("Expert ID | Weight   | Percentage\n")
            f.write("----------|----------|-----------\n")
            for i, w in enumerate(avg_weights):
                f.write(f"Expert {i+1:02d} | {w:.6f} | {w*100:.2f}%\n")
        
        npy_path = os.path.join(folder_path, 'avg_weights.npy')
        np.save(npy_path, avg_weights)

        print(f">> [Success] Weights saved to:\n   1. {txt_path}\n   2. {npy_path}")
        
        print("-" * 50)
        for i, w in enumerate(avg_weights):
            print(f"Expert {i+1}: {w:.4f} ({w*100:.2f}%)")
        print("-" * 50)

    except Exception as e:
        print(f"[Warning] Error in weight analysis: {e}")