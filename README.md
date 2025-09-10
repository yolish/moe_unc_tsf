# MoGU: Mixture-of-Gaussians with Uncertainty-based Gating for Time Series Forecasting

This repository provide the official implementatoin for our paper: "MoGU: Mixture-of-Gaussians with Uncertainty-based Gating for Time Series Forecasting"

This repo implements:
- Stochastic time series forecasting: extending selected TSF models to estimate mean (the prediction) and variance (the uncertainty) 
- Ensemble Mixture-of-Experts for time series forecasting
- Ensemble Mixture-of-Gaussians for time series forecasting
- Ensemble Mixture-of-Gaussians with Uncertainty-based gating (MoGU)
- Uncertainty reporting at inference (Epistermic, Aleatoric and Overall) - per expert and for the entire system

## Setup
```
python -m venv unc_moe
source unc_moe/bin/activate
pip install -r requirements.txt
 pip3 install torch --index-url https://download.pytorch.org/whl/cu128
```

## Datasets
Follow the instructions in TSLib (https://github.com/thuml/Time-Series-Library/) to download and set up the forecasting datasets 

## Training and Inference
Run the following to reproduce training and inference for the main datasets used in the repository
```
bash run_all.sh
```


Our impelemntation leverages the great work of TSLib, providing a general framework for time series analysis:
https://github.com/thuml/Time-Series-Library/


