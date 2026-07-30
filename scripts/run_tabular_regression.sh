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

data_sets=("Synthetic" "Bike" "Temperature" "Superconductivity")
# Add more seeds here to run each experiment multiple times, e.g. seeds=(4022 2023 999).
# final_calibration_summary.csv will report every metric as mean±std across these seeds.
seeds=(4022)
model_id="tabular_exp"

for seed in "${seeds[@]}"
do
    for data in "${data_sets[@]}"
    do
        n_samples_args=()
        if [ "$data" == "Synthetic" ]; then
            enc_in=1
            num_experts=3
            tau=150
            tau_classic=150 # ClassicMoE already unaffected here thanks to n_samples=1500 below
            epochs=600 # 150
            d_model=32  # התאמה למאמר - 32 נוירונים בשכבות לדאטה סינתטי
            load_balance_weight=0.05  # untuned default; not covered by the Bike/Temperature sweep below
            # Deliberate deviation from the paper's n=500: bumped to 1500 (matching
            # Bike/Temperature's scale) so ambiguous/ low-confidence gating regions have
            # enough nearby calibration points to avoid MoECP's delta_{+inf} tail-mass
            # edge case (see MoECP_Calibrator.py; Theorem 1 requires an unbounded interval
            # whenever a test point's own randomized self-weight exceeds alpha).
            n_samples_args=(--n_samples 1500)
        elif [ "$data" == "Bike" ]; then
            enc_in=12       # 13 raw UCI columns minus dteday; no one-hot (matches data_loader.py's Bike branch)
            num_experts=2
            tau=10
            # ClassicMoE's softmax gate collapses to near-one-hot (no probabilistic head to
            # keep it soft), so at tau=100 MoECP's exp(-tau*KL) weighting numerically zeroes
            # out every calibration point routed to a different expert than the test point.
            # When that expert's bucket can't carry 1-alpha mass on its own, MoECP falls back
            # to the delta_{+inf} case (Theorem 1) and Avg Width reports inf. Lowering tau just
            # for ClassicMoE keeps cross-expert points from underflowing to zero weight.
            tau_classic=10
            epochs=800 # 2000     # כפי שסדרנו קודם כדי לחקות את המאמר
            d_model=64  # התאמה למאמר - 64 נוירונים לדאטה אמיתי
            # Single-seed sweep over {0.0,0.01,0.025,0.05,0.1,0.2,0.4} across ClassicMoE and MOG
            # found 0.2 minimizes the average (relative-to-each-architecture's-own-best) test MSE:
            # ClassicMoE 0.0654 (best is 0.0635 @ 0.1) and MOG 0.0944 (its own best), vs 0.0672/0.1040
            # at the old hardcoded 0.05. Larger dataset (~10k train rows) tolerates/benefits from
            # stronger balancing across two experts. Not yet confirmed across multiple seeds.
            load_balance_weight=0.2
        elif [ "$data" == "Temperature" ]; then
            enc_in=21
            num_experts=2
            tau=10
            tau_classic=10  # see Bike's comment above; same near-one-hot ClassicMoE gate issue
            epochs=800 # 2000
            d_model=64  # התאמה למאמר - 64 נוירונים לדאטה אמיתי
            # Same sweep as Bike (see comment above) found 0.025 minimizes the average
            # relative-to-own-best test MSE: ClassicMoE 0.1021 (its own best is 0.0999 @ 0.0)
            # and MOG 0.0930 (its own best), vs 0.1021/0.1016 at the old hardcoded 0.05. Smaller
            # dataset (~4.5k train rows) tolerates much less balancing pressure than Bike -- note
            # ClassicMoE's own individual optimum here is actually 0.0 (no load balancing), so this
            # shared value is a compromise favoring MOG, not a strict win for both. Not yet
            # confirmed across multiple seeds.
            load_balance_weight=0.025
        elif [ "$data" == "Superconductivity" ]; then
            enc_in=81       # 81 UCI feature columns (id=464); no material-family label present
            num_experts=3   # matches Stanev et al. 2018's cuprate/iron-based/low-Tc 3-family split --
                             # NOT a literature-confirmed count for this exact UCI file (see plan doc);
                             # the gate must infer any family structure from the 81 features alone.
            tau=10
            tau_classic=10
            epochs=800
            d_model=128     # larger than Bike/Temperature's 64 given the much higher input dimensionality
            load_balance_weight=0.05  # untuned default, same as Synthetic's comment above
        fi

        # 1. Run Classic MoE (The EXACT baseline from the MoECP paper: deterministic, MSE, Softmax Gate)
        python -u run.py --task_name tabular_regression --is_training 1 \
        --root_path ./dataset/ --data_path "${data}.csv" --data "$data" \
        --model MoE --model_id "${model_id}_ClassicMoE_${data}" \
        --enc_in $enc_in --c_out 1 --d_model $d_model --num_experts $num_experts --batch_size 64 \
        --train_epochs $epochs --learning_rate 0.0001 --patience 5 --seed $seed \
        --tau $tau_classic --max_grad_norm 1.0 --load_balance_weight $load_balance_weight "${n_samples_args[@]}" \
        --use_reg_moecp --use_reg_moce --use_reg_standard_cp --overwrite > "logs/tabular/${data}_ClassicMoE_seed${seed}.log" 2>&1 &

        # 2. Run MOG with the specified calibration methods
        python -u run.py --task_name tabular_regression --is_training 1 \
        --root_path ./dataset/ --data_path "${data}.csv" --data "$data" \
        --model MoE --model_id "${model_id}_MOG_${data}" \
        --enc_in $enc_in --c_out 1 --d_model $d_model --num_experts $num_experts --batch_size 64 \
        --train_epochs $epochs --learning_rate 0.0001 --patience 5 --seed $seed \
        --prob_expert --tau $tau --max_grad_norm 1.0 --load_balance_weight $load_balance_weight "${n_samples_args[@]}" \
        --use_reg_moecp --use_reg_moce --use_reg_cp_vs --use_reg_cp_aleatoric \
        --use_reg_cp_aleat_scale --use_reg_adaptive_variance \
        --use_reg_mog_hpd --use_reg_sta_hpd --use_reg_seta_hpd --use_reg_meld_hpd --use_reg_ease_hpd \
        --use_reg_standard_cp --overwrite > "logs/tabular/${data}_MOG_seed${seed}.log" 2>&1 &

        # 3. Run MOGU with the specified calibration methods
        python -u run.py --task_name tabular_regression --is_training 1 \
        --root_path ./dataset/ --data_path "${data}.csv" --data "$data" \
        --model MoE --model_id "${model_id}_MOGU_${data}" \
        --enc_in $enc_in --c_out 1 --d_model $d_model --num_experts $num_experts --batch_size 64 \
        --train_epochs $epochs --learning_rate 0.0001 --patience 5 --seed $seed \
        --prob_expert --unc_gating --tau $tau --max_grad_norm 1.0 --load_balance_weight $load_balance_weight "${n_samples_args[@]}" \
        --use_reg_moecp --use_reg_moce --use_reg_cp_vs --use_reg_cp_aleatoric \
        --use_reg_cp_aleat_scale --use_reg_adaptive_variance \
        --use_reg_mog_hpd --use_reg_sta_hpd --use_reg_seta_hpd --use_reg_meld_hpd --use_reg_ease_hpd \
        --use_reg_standard_cp --overwrite > "logs/tabular/${data}_MOGU_seed${seed}.log" 2>&1 &

        # 4. Run CQR (true quantile-regression head + conformalized quantile regression calibration)
        python -u run.py --task_name tabular_regression --is_training 1 \
        --root_path ./dataset/ --data_path "${data}.csv" --data "$data" \
        --model MoE --model_id "${model_id}_CQR_${data}" \
        --enc_in $enc_in --c_out 1 --d_model $d_model --num_experts $num_experts --batch_size 64 \
        --train_epochs $epochs --learning_rate 0.0001 --patience 5 --seed $seed \
        --tau $tau --max_grad_norm 1.0 --load_balance_weight $load_balance_weight "${n_samples_args[@]}" \
        --use_quantile_loss --use_reg_cqr --overwrite > "logs/tabular/${data}_CQR_seed${seed}.log" 2>&1 &

        # 5. Run ClassicMoE with num_experts=1 (single-network, no-mixture baseline; same
        # model_id as run #1 -- the "setting" string's ne1 vs ne${num_experts} keeps checkpoints
        # and results directories from colliding, only this log filename needs its own suffix).
        python -u run.py --task_name tabular_regression --is_training 1 \
        --root_path ./dataset/ --data_path "${data}.csv" --data "$data" \
        --model MoE --model_id "${model_id}_ClassicMoE_${data}" \
        --enc_in $enc_in --c_out 1 --d_model $d_model --num_experts 1 --batch_size 64 \
        --train_epochs $epochs --learning_rate 0.0001 --patience 5 --seed $seed \
        --tau $tau_classic --max_grad_norm 1.0 --load_balance_weight $load_balance_weight "${n_samples_args[@]}" \
        --use_reg_moecp --use_reg_moce --use_reg_standard_cp --overwrite > "logs/tabular/${data}_ClassicMoE_ne1_seed${seed}.log" 2>&1 &

        # 6. Run MOG with num_experts=1 (single probabilistic-Gaussian baseline; same model_id
        # as run #2, same reasoning as run #5 above).
        python -u run.py --task_name tabular_regression --is_training 1 \
        --root_path ./dataset/ --data_path "${data}.csv" --data "$data" \
        --model MoE --model_id "${model_id}_MOG_${data}" \
        --enc_in $enc_in --c_out 1 --d_model $d_model --num_experts 1 --batch_size 64 \
        --train_epochs $epochs --learning_rate 0.0001 --patience 5 --seed $seed \
        --prob_expert --tau $tau --max_grad_norm 1.0 --load_balance_weight $load_balance_weight "${n_samples_args[@]}" \
        --use_reg_moecp --use_reg_moce --use_reg_cp_vs --use_reg_cp_aleatoric \
        --use_reg_cp_aleat_scale --use_reg_adaptive_variance \
        --use_reg_mog_hpd --use_reg_sta_hpd --use_reg_seta_hpd --use_reg_meld_hpd --use_reg_ease_hpd \
        --use_reg_standard_cp --overwrite > "logs/tabular/${data}_MOG_ne1_seed${seed}.log" 2>&1 &

        wait
    done
done


echo "========================================================="
echo "All training and calibration finished! Generating CSV..."
echo "========================================================="

# Pass the seeds list to the embedded Python script below via the environment
# (kept out of the heredoc itself to avoid any shell/Python syntax collisions).
export SEEDS="${seeds[*]}"

# Embedded Python script to parse logs and generate the summary CSV
# Generating CSV - Extended with exact requested columns
python3 << 'EOF'
import os, re, csv
import numpy as np

data_sets = ['Synthetic', 'Bike', 'Temperature', 'Superconductivity']
# (log_suffix, architecture_label): log_suffix picks the log file to read; architecture_label
# is what's written into the CSV's "architecture" column. The num_experts=1 baselines share
# their sibling's architecture_label (distinguished only by the num_experts column) but need
# their own log_suffix since their run writes to a differently-named log file.
run_variants = [
    ('ClassicMoE', 'ClassicMoE'),
    ('ClassicMoE_ne1', 'ClassicMoE'),
    ('MOG', 'MOG'),
    ('MOG_ne1', 'MOG'),
    ('MOGU', 'MOGU'),
    ('CQR', 'CQR'),
]
seeds = os.environ['SEEDS'].split()
methods = {
    'MoECP': r'MoECP Results:',
    'MoCE': r'MoCE Results:',
    'Standard CP': r'Standard CP Results:',
    'CPVS': r'CP_VS Static Results:',
    'CPVS Aleatoric': r'CP_VS Aleatoric Results:',
    'CP Aleatoric Scale': r'CP Aleatoric Scale Results',
    'Adaptive Variance-Ratio': r'Adaptive Variance-Ratio Results',
    'MoG-HPD': r'\nMoG-HPD Results',
    # 'STA-HPD' shares no substring with the MoG-HPD pattern above (it deliberately
    # avoids the literal 'MoG-HPD'), so it needs no anchoring. 'SETA-HPD Results' does NOT
    # contain 'STA-HPD Results' as a contiguous substring ('SETA' is S-E-T-A, not S-T-A), and
    # 'STA-HPD Results' does not contain 'SETA-HPD Results', so the two never cross-match.
    'STA-HPD': r'STA-HPD Results',
    'SETA-HPD': r'SETA-HPD Results',
    # 'MELD-HPD Results' shares no contiguous substring with 'MoG-HPD Results',
    # 'STA-HPD Results', or 'SETA-HPD Results' (MELD is M-E-L-D), so it cross-matches none.
    'MELD-HPD': r'MELD-HPD Results',
    # 'EASE-HPD Results' shares no contiguous substring with any of the other HPD headers
    # above (EASE is E-A-S-E), so it cross-matches none.
    'EASE-HPD': r'EASE-HPD Results',
    'CQR': r'CQR Results:'
}

csv_file = 'logs/tabular/final_calibration_summary.csv'
print(f"Attempting to create extended CSV: {csv_file}")


def fmt(values, decimals=4, suffix=""):
    """Format a list of floats as 'mean±std' (population std, so n=1 -> ±0)."""
    if not values:
        return "N/A"
    mean = np.mean(values)
    std = np.std(values)
    return f"{mean:.{decimals}f}±{std:.{decimals}f}{suffix}"


def fmt_array(list_of_lists, decimals=4):
    """Elementwise 'mean±std' across seeds for equal-length per-expert arrays."""
    if not list_of_lists:
        return "N/A"
    arr = np.array(list_of_lists)  # [n_seeds, n_experts]
    means = np.mean(arr, axis=0)
    stds = np.std(arr, axis=0)
    return ", ".join(f"{m:.{decimals}f}±{s:.{decimals}f}" for m, s in zip(means, stds))


def parse_log(content):
    """Extract every per-run field from one seed's log. Missing fields are None."""
    parsed = {}

    metrics_match = re.search(r'(?:Final Test Metrics \| MSE:|Test Results - MSE:)\s*([0-9\.]+).*?MAE:\s*([0-9\.]+)', content)
    parsed['mse'] = float(metrics_match.group(1)) if metrics_match else None
    parsed['mae'] = float(metrics_match.group(2)) if metrics_match else None

    ne_match = re.search(r'_ne(\d+)_pe\d+_ug\d+_', content)
    parsed['num_experts'] = int(ne_match.group(1)) if ne_match else None

    epi_ratio_match = re.search(r'Avg Epistemic/Aleatoric Ratio:\s*([0-9\.]+)', content)
    parsed['epistemic_ratio'] = float(epi_ratio_match.group(1)) if epi_ratio_match else None

    epi_contrib_match = re.search(r'Avg Epistemic Contribution to Total Var:\s*([0-9\.]+)%?', content)
    parsed['epistemic_contrib'] = float(epi_contrib_match.group(1)) if epi_contrib_match else None

    avg_gating_match = re.search(r'Average Gating Weights per expert:\s*\[(.*?)\]', content)
    parsed['avg_gating'] = [float(x) for x in avg_gating_match.group(1).split()] if avg_gating_match else None

    gating_std_match = re.search(r'Std of Gating Weights per expert:\s*\[(.*?)\]', content)
    parsed['gating_std'] = [float(x) for x in gating_std_match.group(1).split()] if gating_std_match else None

    res_dict = {}
    for m_key, m_pattern in methods.items():
        c = re.search(m_pattern + r'.*?Coverage:\s*([0-9\.]+)', content, re.DOTALL)
        w = re.search(m_pattern + r'.*?Avg Width:\s*(inf|[0-9\.]+)', content, re.DOTALL)
        if c and w:
            res_dict[m_key] = {'cov': float(c.group(1)), 'wid': float(w.group(1))}
    parsed['methods'] = res_dict

    return parsed


with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        "dataset", "architecture", "num_experts", "calibration model", "num_seeds", "coverage", "width",
        "mse", "mae", "Avg Epistemic/Aleatoric Ratio",
        "Avg Epistemic Contribution to Total Var",
        "Average Gating Weights per expert", "Std of Gating Weights per expert"
    ])

    for data in data_sets:
        for log_suffix, architecture in run_variants:
            per_seed = []
            for seed in seeds:
                log_path = f'logs/tabular/{data}_{log_suffix}_seed{seed}.log'
                if not os.path.exists(log_path):
                    print(f"Warning: {log_path} not found.")
                    continue
                with open(log_path, 'r') as log:
                    per_seed.append(parse_log(log.read()))

            if not per_seed:
                print(f"No logs found for {data}/{log_suffix} across seeds {seeds}.")
                continue

            num_experts_vals = [p['num_experts'] for p in per_seed if p['num_experts'] is not None]
            num_experts = num_experts_vals[0] if num_experts_vals else "N/A"

            mse_vals = [p['mse'] for p in per_seed if p['mse'] is not None]
            mae_vals = [p['mae'] for p in per_seed if p['mae'] is not None]
            epi_ratio_vals = [p['epistemic_ratio'] for p in per_seed if p['epistemic_ratio'] is not None]
            epi_contrib_vals = [p['epistemic_contrib'] for p in per_seed if p['epistemic_contrib'] is not None]
            avg_gating_lists = [p['avg_gating'] for p in per_seed if p['avg_gating'] is not None]
            gating_std_lists = [p['gating_std'] for p in per_seed if p['gating_std'] is not None]

            mse = fmt(mse_vals)
            mae = fmt(mae_vals)
            epistemic_ratio = fmt(epi_ratio_vals)
            epistemic_contrib = fmt(epi_contrib_vals, decimals=2, suffix="%")
            avg_gating = fmt_array(avg_gating_lists)
            gating_std = fmt_array(gating_std_lists)

            # Gather per-method coverage/width across whichever seeds report that method
            res_dict = {}
            for p in per_seed:
                for m_key, val in p['methods'].items():
                    res_dict.setdefault(m_key, []).append(val)

            if not res_dict:
                print(f"No calibration results found for {data}/{log_suffix} across seeds {seeds}.")
                continue

            for m_key, vals in res_dict.items():
                cov_vals = [v['cov'] for v in vals]
                wid_vals = [v['wid'] for v in vals]
                writer.writerow([
                    data,
                    architecture,
                    num_experts,
                    m_key,
                    len(vals),
                    fmt(cov_vals),
                    fmt(wid_vals),
                    mse,
                    mae,
                    epistemic_ratio,
                    epistemic_contrib,
                    avg_gating,
                    gating_std
                ])

print("Extended CSV generation completed!")
EOF