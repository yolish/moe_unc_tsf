# Uncertainty-Aware Mixture-of-Experts for Time Series Forecasting

## WORK IN PROGRESS 

This repository provide the official implementatoin for Uncertainty-Aware Mixture-of-Experts

Main contributions in the repo:
- We extend existing time series forecasting with Ensemble MoE
- We extend TSF models as probabilistic models
- We enable MoE with probabilistic experts, yielding both per expert and overall uncertainty estimation 
- We introduce uncertainty driven gating

## Setup
```
python -m venv unc_moe
source unc_moe/bin/activate
pip install -r requirements.txt
 pip3 install torch --index-url https://download.pytorch.org/whl/cu128
```

## Training

## Infernce


Our impelemntation leverages the great work of TSLib, providing a general framework for time series analysis:
https://github.com/thuml/Time-Series-Library/


