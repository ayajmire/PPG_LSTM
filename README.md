# Blood Pressure Estimation LSTM from Camera

An LSTM model for estimating systolic and diastolic blood pressure from PPG (Photoplethysmography) waveforms.

## Overview
This repository contains a deep learning pipeline to:
- Preprocess raw PPG waveform data
- Train an LSTM model for blood pressure regression
- Convert the model to CoreML for iOS deployment
- Evaluate model performance using external data

## Contents
- `PPG_LSTM_Model.py` — PyTorch LSTM model for SBP/DBP prediction
- `PPG_preprocess_external_data.py` — Preprocessing script for evaluation datasets
- `lstm_bp_model.pth` — Trained PyTorch model weights
- `pytorch2torchml.py` — PyTorch to CoreML conversion script
- `PPG_BP_Estimator.mlpackage.zip` — Exported CoreML model
- `dbp_chart.png` and `sbp_chart.png` — Model evaluation visualizations
- `ExternalValidation.py` — Code to validate on a separate dataset
- `PPG-BP dataset.xlsx` — Dataset reference or sample metadata

## Results
The LSTM achieved a MAE of ~5 mmHg on SBP and ~2 mmHg on DBP using external validation data.

## Future Work
- Integrate with iOS camera feed for real-time PPG capture
- Expand dataset and improve generalization
