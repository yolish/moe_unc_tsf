import os
import numpy as np
import pandas as pd
import re

def parse_run_name(run_name):
    """
    Extracts parameters from the run folder name using regular expressions
    and maps them to the relevant architecture.
    """
    params = {
        'run_name': run_name,
        'base_model': 'Unknown',
        'dataset': 'Unknown',
        'architecture': 'Unknown',
        'num_experts': 1,
        'horizon': 0,
        'seed': -1
    }
    
    try:
        parts = run_name.split('_')
        
        ne_idx = -1
        for i, p in enumerate(parts):
            if p.startswith('ne') and p[2:].isdigit():
                ne_idx = i
                break
                
        if ne_idx >= 4:
            params['base_model'] = parts[ne_idx-2]
            params['dataset'] = parts[ne_idx-1]
        
        ne_match = re.search(r'_ne(\d+)_', run_name)
        if ne_match: 
            params['num_experts'] = int(ne_match.group(1))
            
        pe_match = re.search(r'_(?:pe|pmo)(\d+)_', run_name)
        prob_expert = int(pe_match.group(1)) if pe_match else 0
        
        ug_match = re.search(r'_ug(\d+)_', run_name)
        unc_gating = int(ug_match.group(1)) if ug_match else 0
        
        pl_match = re.search(r'_pl(\d+)_', run_name)
        if pl_match: 
            params['horizon'] = int(pl_match.group(1))
            
        seed_match = re.search(r'_seed(\d+)', run_name)
        if seed_match: 
            params['seed'] = int(seed_match.group(1))
            
        if params['num_experts'] > 1:
            if unc_gating == 1:
                params['architecture'] = 'MoGU'
            elif prob_expert == 1:
                params['architecture'] = 'MoG'
            else:
                params['architecture'] = 'MoE'
        else:
            params['architecture'] = 'Baseline'
            
    except Exception as e:
        print(f"Error parsing {run_name}: {e}")
        
    return params

def collect_results(results_dir):
    data = []
    
    if not os.path.exists(results_dir):
        print(f"Directory {results_dir} does not exist.")
        return pd.DataFrame()
        
    for folder in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, folder)
        metrics_path = os.path.join(folder_path, 'metrics.npy')
        
        if os.path.isdir(folder_path) and os.path.exists(metrics_path):
            # [mae, mse, rmse, mape, mspe]
            metrics = np.load(metrics_path)
            mae = metrics[0]
            mse = metrics[1]
            
            params = parse_run_name(folder)
            params['mae'] = mae
            params['mse'] = mse
            
            data.append(params)
            
    return pd.DataFrame(data)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    results_dir = os.path.join(project_root, 'results')
    output_csv = os.path.join(project_root, 'parsed_results_summary.csv')
    
    df = collect_results(results_dir)
    
    if not df.empty:
        cols = ['architecture', 'base_model', 'dataset', 'horizon', 'num_experts', 'seed', 'mse', 'mae', 'run_name']
        df = df[cols]
        
        df.to_csv(output_csv, index=False)
        print(f"Successfully parsed results.")
        print(f"File saved to: {output_csv}\n")
        print("Preview:")
        print(df.head())
    else:
        print("No results found.")