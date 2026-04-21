import re
import pandas as pd
import os

log_file = "calibration_full_results.log"
results = []

if os.path.exists(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()

    current_dataset, current_method, current_model_type, current_model_name, current_seed, current_feat, current_ne = [None]*7

    for line in lines:
        if "--- Running" in line:
            current_model_type = "MOG" if "MOG (config 2)" in line else "MoGU"
            ds_match = re.search(r"for ([\w_]+)\.csv \(Seed: (\d+), Feat: (\w+), Model: (\w+), Experts: (\d+)\)", line)
            if ds_match: 
                current_dataset = ds_match.group(1).replace('national_illness', 'ILI')
                current_seed = ds_match.group(2)
                current_feat = ds_match.group(3)
                current_model_name = ds_match.group(4)
                current_ne = ds_match.group(5)

        if "Running CP calibration" in line: current_method = "Standard CP"
        elif "Running CPVS calibration" in line or "cpvs" in line.lower(): current_method = "CPVS"
        elif "Aleatoric MOG calibration" in line: current_method = "Aleatoric MOG"
        elif "Aleatoric Only calibration" in line: current_method = "Aleatoric Only"
        
        cov_match = re.search(r"[Cc]overage\s*[:=]\s*([\d\.]+)", line)
        width_match = re.search(r"[Ww]idth\s*[:=]\s*([\d\.]+)", line)
        
        if cov_match and width_match and current_dataset and current_method:
            results.append({
                "Dataset": current_dataset, 
                "BaseModel": current_model_name,
                "Seed": current_seed, 
                "Features": current_feat,
                "Experts": current_ne,
                "Type": current_model_type,
                "Method": current_method,
                "Value": f"{float(cov_match.group(1)):.4f} / {float(width_match.group(1)):.4f}"
            })
            current_method = None 

if results:
    df = pd.DataFrame(results).drop_duplicates()
    pivot_df = df.pivot(index=["Dataset", "BaseModel", "Seed", "Features", "Experts", "Type"], columns="Method", values="Value")
    pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)
    
    print("\n" + pivot_df.to_markdown())
    pivot_df.to_csv("calibration_summary_final.csv")
    print("\nSummary saved to: calibration_summary_final.csv")
