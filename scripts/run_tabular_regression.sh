#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1

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
        elif [ "$data" == "Bike" ]; then
            enc_in=12
            num_experts=2
            tau=100
            epochs=2000
        elif [ "$data" == "Temperature" ]; then
            enc_in=21
            num_experts=2
            tau=100
            epochs=500
        fi

        # Run MOG with the 4 specified calibration methods
        python -u run.py --task_name tabular_regression --is_training 0 \
        --root_path ./dataset/ --data_path "${data}.csv" --data "$data" \
        --model MoE --model_id "${model_id}_MOG_${data}" \
        --enc_in $enc_in --c_out 1 --num_experts $num_experts --batch_size 64 \
        --train_epochs $epochs --learning_rate 0.0001 --patience 5 --seed $seed \
        --prob_expert --tau $tau \
        --use_reg_moecp --use_reg_cp_vs --use_reg_cp_aleatoric \
        --use_reg_cp_aleat_scale > "logs/tabular/${data}_MOG.log" 2>&1 &

        # Run MOGU with the 4 specified calibration methods
        python -u run.py --task_name tabular_regression --is_training 0 \
        --root_path ./dataset/ --data_path "${data}.csv" --data "$data" \
        --model MoE --model_id "${model_id}_MOGU_${data}" \
        --enc_in $enc_in --c_out 1 --num_experts $num_experts --batch_size 64 \
        --train_epochs $epochs --learning_rate 0.0001 --patience 5 --seed $seed \
        --prob_expert --unc_gating --tau $tau \
        --use_reg_moecp --use_reg_cp_vs --use_reg_cp_aleatoric \
        --use_reg_cp_aleat_scale > "logs/tabular/${data}_MOGU.log" 2>&1 &
        
        wait
    done
done

echo "========================================================="
echo "All training and calibration finished! Generating CSV..."
echo "========================================================="

# Embedded Python script to parse logs and generate the summary CSV
python3 << 'EOF'
import os, re

datasets = ['Synthetic', 'Bike', 'Temperature']
models = ['MOG', 'MOGU']

csv_lines = []
csv_lines.append("Dataset,Backbone,Calibration Method,Coverage,Avg Width,Is_Best")

# Mapping the 4 selected calibration methods to their respective log signatures
methods = {
    'MoECP': r'MoECP Results:',
    'CPVS': r'CP_VS Static Results:',
    'CPVS Aleatoric': r'CP_VS Aleatoric Results:',
    'CP Aleatoric Scale': r'CP Aleatoric Scale Results'
}

for data in datasets:
    for model in models:
        log_file = f'logs/tabular/{data}_{model}.log'
        if not os.path.exists(log_file):
            continue
            
        with open(log_file, 'r') as f:
            content = f.read()
            
        results = {}
        for m_name, m_str in methods.items():
            cov_match = re.search(m_str + r'.*?Coverage:\s*([0-9\.]+)', content, re.DOTALL)
            wid_match = re.search(m_str + r'.*?Width:\s*([0-9\.]+)', content, re.DOTALL)
            if cov_match and wid_match:
                results[m_name] = {
                    'cov': float(cov_match.group(1)),
                    'wid': float(wid_match.group(1))
                }
                
        if not results:
            continue
            
        # Determine the best method based on valid coverage (>= 90%) and minimum width
        valid_methods = {k: v for k, v in results.items() if v['cov'] >= 0.895}
        if valid_methods:
            best_method = min(valid_methods.items(), key=lambda x: x[1]['wid'])[0]
        else:
            best_method = max(results.items(), key=lambda x: x[1]['cov'])[0]
            
        for m_name, res in results.items():
            cov_str = f"{res['cov']:.4f}"
            wid_str = f"{res['wid']:.4f}"
            is_best = "Yes" if m_name == best_method else "No"
            
            csv_lines.append(f"{data},{model},{m_name},{cov_str},{wid_str},{is_best}")

output_text = '\n'.join(csv_lines)

# Save the final table to a CSV file
csv_path = 'logs/tabular/final_calibration_summary.csv'
with open(csv_path, 'w') as f:
    f.write(output_text)
    
print(output_text)
print(f"\n[!] The CSV file has been saved to: {csv_path}")
EOF