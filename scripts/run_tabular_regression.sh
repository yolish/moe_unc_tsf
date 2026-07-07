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
            d_model=64
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
# Generating CSV - Extended with MoE Analysis
python3 << 'EOF'
import os, re, csv

data_sets = ['Synthetic', 'Bike', 'Temperature']
# הוספנו את ClassicMoE כדי שהפייתון יחלץ גם את תוצאות הרפרנס
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
    # הוספנו את העמודות MSE ו-MAE יחד עם מדדי ה-MoE
    writer.writerow(["Dataset", "Backbone", "Calibration", "Coverage", "Width", "MSE", "MAE", "Is_Best", 
                     "Epistemic_Ratio", "Gating_Std", "Expert_Win_Dist"])

    for data in data_sets:
        for model in models:
            log_path = f'logs/tabular/{data}_{model}.log'
            if not os.path.exists(log_path):
                print(f"Warning: {log_path} not found.")
                continue
            
            with open(log_path, 'r') as log:
                content = log.read()
            
            # --- שליפת ה-MSE וה-MAE מהלוג ---
            # תופס גם הדפסה ישנה וגם חדשה, שולף את שני המספרים בצורה בטוחה
            metrics_match = re.search(r'(?:Final Test Metrics \| MSE:|Test Results - MSE:)\s*([0-9\.]+).*?MAE:\s*([0-9\.]+)', content)
            if metrics_match:
                mse = metrics_match.group(1)
                mae = metrics_match.group(2)
            else:
                mse = "N/A"
                mae = "N/A"

            # --- חילוץ מדדי ה-MoE מהלוג ---
            # 1. יחס השונות האפיסטמית
            epi_ratio_match = re.search(r'Avg Epistemic/Aleatoric Ratio:\s*([0-9\.]+)', content)
            epistemic_ratio = epi_ratio_match.group(1) if epi_ratio_match else "N/A"
            
            # 2. סטיית תקן של הניתוב (מחיקת רווחים כפולים כדי שייראה טוב ב-CSV)
            gating_std_match = re.search(r'Std of Gating Weights per expert:\s*\[(.*?)\]', content)
            if gating_std_match:
                # הופך "[0.15   0.20]" ל-"0.15, 0.20"
                gating_std = re.sub(r'\s+', ', ', gating_std_match.group(1).strip())
            else:
                gating_std = "N/A"
                
            # 3. חילוץ חלוקת המנצחים (אחוזים בלבד)
            experts_matches = re.findall(r'Expert\s+\d+:\s+\d+\s+samples\s+\(([\d\.]+)%\)', content)
            if experts_matches:
                # מייצר מחרוזת בסגנון: "E0: 60.5%, E1: 39.5%"
                win_dist = ", ".join([f"E{i}: {pct}%" for i, pct in enumerate(experts_matches)])
            else:
                win_dist = "N/A"
            # -------------------------------
                
            res_dict = {}
            for m_key, m_pattern in methods.items():
                c = re.search(m_pattern + r'.*?Coverage:\s*([0-9\.]+)', content, re.DOTALL)
                w = re.search(m_pattern + r'.*?Width:\s*([0-9\.]+)', content, re.DOTALL)
                if c and w:
                    res_dict[m_key] = {'cov': float(c.group(1)), 'wid': float(w.group(1))}
            
            if not res_dict:
                print(f"No calibration results found in {log_path}")
                continue
            
            valid = {k: v for k, v in res_dict.items() if v['cov'] >= 0.89}
            best = min(valid.items(), key=lambda x: x[1]['wid'])[0] if valid else max(res_dict.items(), key=lambda x: x[1]['cov'])[0]
            
            for m_key, val in res_dict.items():
                # הוספתי כאן את ה-mse וה-mae לתוך מערך הכתיבה! 
                writer.writerow([
                    data, model, m_key, f"{val['cov']:.4f}", f"{val['wid']:.4f}", 
                    mse, mae, "Yes" if m_key == best else "No",
                    epistemic_ratio, gating_std, win_dist
                ])
                
print("Extended CSV generation completed!")
EOF
# ------------------------------------