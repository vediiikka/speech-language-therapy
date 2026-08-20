# Reproducibility Summary

This document lists all configuration details, scripts, and artifact paths required to reproduce the Speech Emotion Recognition (SER) / Clinical Affect Detection pipeline.

## 1. Dataset Split
- **Dataset**: RAVDESS (432 WAV files, balanced across 6 mapped clinical affect classes).
- **Split Strategy**: Strict speaker-independent partition by actor directories.
  - **Train Set (66.7%)**: `Actor_01` to `Actor_16` (288 samples)
  - **Validation Set (16.7%)**: `Actor_17` to `Actor_20` (72 samples)
  - **Test Set (16.7%)**: `Actor_21` to `Actor_24` (72 samples)

## 2. Preprocessing
- **Resampling**: Standardized to **16,000 Hz** mono PCM.
- **Fixed Window**: Padded or truncated to **3.5 seconds** (56,000 samples).
- **Waveform Normalization**: Fixed to zero-mean unit-variance scaling for Wav2Vec2 compatibility.
- **Classical Feature Scaling**: Standard scaler fit on training split only, saved to `artifacts/baseline/scaler.joblib`.

## 3. Models Evaluated
- **Classical Baselines**:
  - Logistic Regression (balanced class weights)
  - Support Vector Machine (RBF kernel, balanced class weights, probability=True)
  - Random Forest (200 estimators, balanced class weights)
- **Deep Learning Model**:
  - Wav2Vec2 (`facebook/wav2vec2-base-960h` backbone with a 2-layer classification head)

## 4. Key Metrics
- **SVM (Final Production Model)**:
  - Validation Accuracy: `0.4861`
  - Validation Macro-F1: `0.4822`
  - Test Accuracy: `0.3472`
  - Test Macro-F1: `0.3348`
- **Wav2Vec2 (Experimentally Rejected)**:
  - Validation Accuracy: `0.2083`
  - Validation Macro-F1: `0.0984`
  - Test Accuracy: `0.1528`
  - Test Macro-F1: `0.0625`

## 5. Important Scripts
- **Baseline Training**: [`train_baseline.py`](file:///c:/Users/Dell/Downloads/ml_shareable/ml/train_baseline.py)
- **Baseline Evaluation**: [`evaluate_baseline.py`](file:///c:/Users/Dell/Downloads/ml_shareable/ml/evaluate_baseline.py)
- **Wav2Vec2 Training**: [`train_wav2vec2.py`](file:///c:/Users/Dell/Downloads/ml_shareable/ml/train_wav2vec2.py)
- **Wav2Vec2 Evaluation**: [`evaluate_wav2vec2.py`](file:///c:/Users/Dell/Downloads/ml_shareable/ml/evaluate_wav2vec2.py)
- **Production Inference**: [`predict_baseline.py`](file:///c:/Users/Dell/Downloads/ml_shareable/ml/predict_baseline.py)

## 6. Important Artifact Paths
- **Scaler**: `artifacts/baseline/scaler.joblib`
- **Label Encoder**: `artifacts/baseline/label_encoder.joblib`
- **SVM Model Weight**: `artifacts/baseline/best_model.joblib`
- **Classical Feature Config**: `artifacts/baseline/feature_config.json`
- **Classical Extracted Features**: `artifacts/baseline/features.npz`
- **Wav2Vec2 Retrained Checkpoint**: `artifacts/wav2vec2_retrained/best_model.pt`
- **Wav2Vec2 Retrained Config**: `artifacts/wav2vec2_retrained/config.json`

## 7. Production Model Selection
- **Selected Model**: `SVM`
- **Inference Script**: [`predict_baseline.py`](file:///c:/Users/Dell/Downloads/ml_shareable/ml/predict_baseline.py)
