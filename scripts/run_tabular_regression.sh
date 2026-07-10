#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1

echo "Cleaning up old logs in logs/tabular/..."
rm -f ./logs/tabular/*.log
# ------------------------------------
rm -f ./logs/tabular/final_calibration_summary.csv

mkdir -p ./logs/tabular

if [ ! -d "./logs/tabular" ]; then
    mkdir -p ./logs/tabular
fi

data_sets=("Synthetic" "Bike" "Temperature")
seeds=(4021)
model_id="tabular_exp"

for seed in "${seeds[@]}"
do
    for data in "${data_sets[@]}"
    do
        if [ "$data" == "Synthetic" ]; then
            enc_in=1
            num_experts=3
            tau=150
            epochs=600
            d_model=32  # התאמה למאמר - 32 נוירונים בשכבות לדאטה סינתטי
        elif [ "$data" == "Bike" ]; then
            enc_in=60       # <--- שינוי מ-12 ל-60 בגלל הקידוד הקטגוריאלי
            num_experts=2
            tau=100
            epochs=2000     # כפי שסדרנו קודם כדי לחקות את המאמר
            d_model=64  # התאמה למאמר - 64 נוירונים לדאטה אמיתי
        elif [ "$data" == "Temperature" ]; then
            enc_in=21
            num_experts=2
            tau=100
            epochs=500
            d_model=64  # התאמה למאמר - 64 נוירונים לדאטה אמיתי
        fi

        # 1. Run Classic MoE (The EXACT baseline from the MoECP paper: deterministic, MSE, Softmax Gate)
        python -u run.py --task_name tabular_regression --is_training 1 \
        --root_path ./dataset/ --data_path "${data}.csv" --data "$data" \
        --model MoE --model_id "${model_id}_ClassicMoE_${data}" \
        --enc_in $enc_in --c_out 1 --d_model $d_model --num_experts $num_experts --batch_size 64 \
        --train_epochs $epochs --learning_rate 0.0001 --patience 5 --seed $seed \
        --tau $tau \
        --use_reg_moecp --use_reg_standard_cp --overwrite > "logs/tabular/${data}_ClassicMoE.log" 2>&1 &

        # 2. Run MOG with the specified calibration methods
        python -u run.py --task_name tabular_regression --is_training 1 \
        --root_path ./dataset/ --data_path "${data}.csv" --data "$data" \
        --model MoE --model_id "${model_id}_MOG_${data}" \
        --enc_in $enc_in --c_out 1 --d_model $d_model --num_experts $num_experts --batch_size 64 \
        --train_epochs $epochs --learning_rate 0.0001 --patience 5 --seed $seed \
        --prob_expert --tau $tau \
        --use_reg_moecp --use_reg_cp_vs --use_reg_cp_aleatoric \
        --use_reg_cp_aleat_scale --use_reg_standard_cp --overwrite > "logs/tabular/${data}_MOG.log" 2>&1 &

        # 3. Run MOGU with the specified calibration methods
        python -u run.py --task_name tabular_regression --is_training 1 \
        --root_path ./dataset/ --data_path "${data}.csv" --data "$data" \
        --model MoE --model_id "${model_id}_MOGU_${data}" \
        --enc_in $enc_in --c_out 1 --d_model $d_model --num_experts $num_experts --batch_size 64 \
        --train_epochs $epochs --learning_rate 0.0001 --patience 5 --seed $seed \
        --prob_expert --unc_gating --tau $tau \
        --use_reg_moecp --use_reg_cp_vs --use_reg_cp_aleatoric \
        --use_reg_cp_aleat_scale --use_reg_standard_cp --overwrite > "logs/tabular/${data}_MOGU.log" 2>&1 &
        
        wait
    done
done


echo "========================================================="
echo "All training and calibration finished! Generating CSV..."
echo "========================================================="

# Embedded Python script to parse logs and generate the summary CSV
# Generating CSV - Extended with exact requested columns
python3 << 'EOF'
import os, re, csv

data_sets = ['Synthetic', 'Bike', 'Temperature']
models = ['ClassicMoE', 'MOG', 'MOGU']
methods = {
    'MoECP': r'MoECP Results:',
    'Standard CP': r'Standard CP Results:',
    'CPVS': r'CP_VS Static Results:',
    'CPVS Aleatoric': r'CP_VS Aleatoric Results:',
    'CP Aleatoric Scale': r'CP Aleatoric Scale Results'
}

csv_file = 'logs/tabular/final_calibration_summary.csv'
print(f"Attempting to create extended CSV: {csv_file}")

with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    # הגדרת העמודות בדיוק לפי הבקשה
    writer.writerow([
        "dataset", "architecture", "calibration model", "coverage", "width", 
        "mse", "mae", "Avg Epistemic/Aleatoric Ratio", 
        "Avg Epistemic Contribution to Total Var", 
        "Average Gating Weights per expert", "Std of Gating Weights per expert"
    ])

    for data in data_sets:
        for model in models:
            log_path = f'logs/tabular/{data}_{model}.log'
            if not os.path.exists(log_path):
                print(f"Warning: {log_path} not found.")
                continue
            
            with open(log_path, 'r') as log:
                content = log.read()
            
            # --- שליפת ה-MSE וה-MAE מהלוג ---
            metrics_match = re.search(r'(?:Final Test Metrics \| MSE:|Test Results - MSE:)\s*([0-9\.]+).*?MAE:\s*([0-9\.]+)', content)
            if metrics_match:
                mse = metrics_match.group(1)
                mae = metrics_match.group(2)
            else:
                mse = "N/A"
                mae = "N/A"

            # --- שליפת מדדי Uncertainty (אפיסטמי / אליאטורי) ---
            epi_ratio_match = re.search(r'Avg Epistemic/Aleatoric Ratio:\s*([0-9\.]+)', content)
            epistemic_ratio = epi_ratio_match.group(1) if epi_ratio_match else "N/A"
            
            epi_contrib_match = re.search(r'Avg Epistemic Contribution to Total Var:\s*([0-9\.]+%?)', content)
            epistemic_contrib = epi_contrib_match.group(1) if epi_contrib_match else "N/A"

            # --- שליפת מדדי ה-MoE Gating ---
            avg_gating_match = re.search(r'Average Gating Weights per expert:\s*\[(.*?)\]', content)
            if avg_gating_match:
                # הפיכת רווחים לפסיקים כדי שיוצג יפה (לדוגמה: "0.5, 0.5")
                avg_gating = re.sub(r'\s+', ', ', avg_gating_match.group(1).strip())
            else:
                avg_gating = "N/A"

            gating_std_match = re.search(r'Std of Gating Weights per expert:\s*\[(.*?)\]', content)
            if gating_std_match:
                gating_std = re.sub(r'\s+', ', ', gating_std_match.group(1).strip())
            else:
                gating_std = "N/A"

            # --- חילוץ תוצאות הכיול ---
            res_dict = {}
            for m_key, m_pattern in methods.items():
                c = re.search(m_pattern + r'.*?Coverage:\s*([0-9\.]+)', content, re.DOTALL)
                w = re.search(m_pattern + r'.*?Width:\s*([0-9\.]+)', content, re.DOTALL)
                if c and w:
                    res_dict[m_key] = {'cov': float(c.group(1)), 'wid': float(w.group(1))}
            
            if not res_dict:
                print(f"No calibration results found in {log_path}")
                continue
            
            # כתיבת השורות לקובץ ה-CSV
            for m_key, val in res_dict.items():
                writer.writerow([
                    data, 
                    model, 
                    m_key, 
                    f"{val['cov']:.4f}", 
                    f"{val['wid']:.4f}", 
                    mse, 
                    mae, 
                    epistemic_ratio, 
                    epistemic_contrib, 
                    avg_gating, 
                    gating_std
                ])
                
print("Extended CSV generation completed!")
EOF