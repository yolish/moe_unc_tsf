#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
models=("iTransformer" "PatchTST")

# רשימת מערכי הנתונים (4 ETT, ILI, 2 Large, 1 XL)
root_paths=(
  "./data/long_term_forecast/ETT/" "./data/long_term_forecast/ETT/" "./data/long_term_forecast/ETT/" "./data/long_term_forecast/ETT/"
  "./data/long_term_forecast/illness/"
  "./data/long_term_forecast/electricity/" "./data/long_term_forecast/weather/"
  "./data/long_term_forecast/traffic/"
)
data_paths=(
  "ETTh2.csv" "ETTm2.csv" "ETTm1.csv" "ETTh1.csv" 
  "national_illness.csv" 
  "electricity.csv" "weather.csv" 
  "traffic.csv"
) 
datasets=("ETTh2" "ETTm2" "ETTm1" "ETTh1" "custom" "custom" "custom" "custom")

# הגדרות הרצה - pred_len 24 עבור ILI, 96 עבור השאר
pred_lengths=(96 96 96 96 24 96 96 96)
num_experts=(3)
configurations=(2 3) # 2 = MOG, 3 = MoGU

# לולאות פרמטרים
seeds=(4021)
features_list=("S" "MS") # M = multivariate, S = univariate (single feature) 

model_id="test"
seq_len=96
label_len=48
batch_size=8

LOG_FILE="calibration_full_results.log"
> $LOG_FILE

length=${#root_paths[@]}

for seed in "${seeds[@]}"
do
  for features in "${features_list[@]}"
  do
    for model_name in "${models[@]}"
    do
      for ((i=0; i<$length; i++))
      do
        root_path=${root_paths[$i]}
        data_path=${data_paths[$i]}
        dataset=${datasets[$i]}
        pred_len=${pred_lengths[$i]}
        
        # Learning Rate מותאם אישית למערכים הגדולים
        lr_arg=""
        if [[ "$data_path" == "electricity.csv" || "$data_path" == "weather.csv" || "$data_path" == "traffic.csv" ]]; then
            lr_arg="--learning_rate 0.001"
        fi
        
        for ne in "${num_experts[@]}"
        do
          for config in "${configurations[@]}"
          do
              # קביעת סוג המודל להדפסה בלוג (עבור ה-Parsing)
              run_type="MOG"
              unc_flag=""
              if [ $config -eq 3 ]; then
                  run_type="MoGU"
                  unc_flag="--unc_gating"
              fi

              echo "--- Running $run_type (config $config) for $data_path (Seed: $seed, Feat: $features, Model: $model_name, Experts: $ne) ---" | tee -a $LOG_FILE
              
              # דגלי שמירה ל-MoGU באורך 96
              save_flags=""
              if [[ $config -eq 3 && $pred_len -eq 96 ]]; then
                  save_flags="--save_outputs --save_unc --save_expert_outputs"
              fi

              python -u run.py \
              --task_name long_term_forecast \
              --is_training 0 \
              --root_path $root_path \
              --data_path $data_path \
              --model_id $model_id \
              --model $model_name \
              --data $dataset \
              --features $features \
              --seq_len $seq_len \
              --batch_size $batch_size \
              --label_len $label_len \
              --pred_len $pred_len \
              --seed $seed \
              --num_experts $ne \
              --prob_expert \
              --max_grad_norm 1 \
              $unc_flag \
              $lr_arg \
              $save_flags \
              --do_cp_calibration \
              --do_cpvs_calibration \
              --do_aleatoric_mog_calibration \
              --do_aleatoric_mog_calibration_second_option \
              --do_aleatoric_only_calibration 2>&1 | tee -a $LOG_FILE
          done
        done    
      done
    done
  done
done

# עיבוד הלוג לטבלה עם שורות נפרדות ל-MOG ו-MoGU
cat << 'EOF' > parse_log_to_table.py
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
EOF

python parse_log_to_table.py

# פתיחה אוטומטית של הקובץ
if command -v xdg-open > /dev/null; then xdg-open calibration_summary_final.csv;
elif command -v open > /dev/null; then open calibration_summary_final.csv;
elif command -v start > /dev/null; then start calibration_summary_final.csv;
fi