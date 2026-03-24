#!/bin/bash
# run_synthetic_comparison.sh

# ניווט פנימי של הסקריפט אל התיקייה שבה נמצאים קבצי הפייתון
cd load_balancing_analysis/regression_task/

# הגדרת נתיב כדי שפייתון יזהה את התיקיות של הפרויקט (data_provider, models)
export PYTHONPATH=$PYTHONPATH:$(pwd):../..

echo "======================================================"
echo "  Starting Analysis for Dataset A (Two Regimes) "
echo "======================================================"

echo "Training Standard MoE..."
python run_synthetic.py --dataset A --num_experts 2

echo "Training Probabilistic MoE (MoG)..."
python run_synthetic.py --dataset A --num_experts 2 --prob_expert

echo "Training MoGU..."
python run_synthetic.py --dataset A --num_experts 2 --prob_expert --unc_gating

echo ">>> Generating Combined Comparison Graph for Dataset A..."
python run_combined_comparison.py --dataset A --num_experts 2


echo ""
echo "======================================================"
echo "  Starting Analysis for Dataset B (Three Regimes) "
echo "======================================================"

echo "Training Standard MoE..."
python run_synthetic.py --dataset B --num_experts 3

echo "Training Probabilistic MoE (MoG)..."
python run_synthetic.py --dataset B --num_experts 3 --prob_expert

echo "Training MoGU..."
python run_synthetic.py --dataset B --num_experts 3 --prob_expert --unc_gating

echo ">>> Generating Combined Comparison Graph for Dataset B..."
python run_combined_comparison.py --dataset B --num_experts 3


echo "======================================================"
echo "✅ All comparisons and combined graphs generated successfully!"
echo "Check the 'load_balancing_analysis/regression_task/synthetic_results' folder."