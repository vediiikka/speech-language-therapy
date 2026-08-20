# Speech Therapy - ML Emotion & Affect Recognition

This module provides Speech Emotion Recognition (SER) / Clinical Affect Detection for the SIH 2026 Speech Therapy project.

---

## 🛠️ 1. Setup & Installation

### Step 1: Create and activate a virtual environment (Recommended)
`ash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
`

### Step 2: Install dependencies
`ash
pip install -r requirements.txt
`

---

## 🚀 2. How to Run

### A. Run Inference / Prediction on Audio
Run inference on any .wav audio sample:

`ash
# 1. Classical Baseline Predictor (Fast, lightweight)
python predict_baseline.py path/to/sample.wav

# 2. Wav2Vec2 Deep Learning Predictor (Pretrained + Fine-tuned)
python predict_wav2vec2.py path/to/sample.wav
`

*(If no audio path is provided, it automatically tests on a sample from data/raw/ravdess/)*

---

### B. Evaluate Existing Models on Test Set
`ash
# Evaluate Classical Baseline (Logistic Regression, SVM, Random Forest)
python evaluate_baseline.py

# Evaluate Wav2Vec2 Model & Generate Comparison Report
python evaluate_wav2vec2.py
`

---

### C. Retrain the Models (Optional)
`ash
# Train classical baseline models and extract features
python train_baseline.py

# Train / Fine-tune Wav2Vec2 model
python train_wav2vec2.py
`

---

## 📁 3. Project Structure
`	ext
ml/
├── artifacts/              # Saved model checkpoints, scalers & configs
│   ├── baseline/           # Scikit-learn models, features.npz
│   └── wav2vec2/           # Wav2Vec2 PyTorch weights (best_model.pt)
├── data/
│   └── raw/ravdess/        # Audio dataset split by Actor
├── src/
│   ├── dataset.py          # Dataset loaders & audio augmentation
│   ├── feature_extraction.py # MFCC, ZCR, RMS, pitch & spectral features
│   ├── utils.py            # Metrics, seeds, JSON utilities
│   └── models/
│       ├── classical.py    # Baseline classifiers (LR, SVM, RF)
│       └── wav2vec2.py     # Wav2Vec2 classification head architecture
├── config.py               # Central project configuration & dynamic paths
├── requirements.txt        # Python package requirements
├── predict_baseline.py     # CLI inference script (Baseline)
├── predict_wav2vec2.py     # CLI inference script (Wav2Vec2)
├── evaluate_baseline.py    # Test set evaluation (Baseline)
├── evaluate_wav2vec2.py    # Test set evaluation (Wav2Vec2)
├── train_baseline.py       # Training pipeline (Baseline)
└── train_wav2vec2.py       # Training pipeline (Wav2Vec2)
`

---

## 🏆 4. Final Pipeline Results & Production Model

### Dataset & Split
- **Dataset**: Balanced 432 RAVDESS WAV audio samples.
- **Split Strategy**: Strict speaker-independent partition by actor identity (Train: Actor_01–16, Val: Actor_17–20, Test: Actor_21–24).

### Model Performance Summary

| Model | Val Accuracy | Val Macro-F1 | Test Accuracy | Test Macro-F1 | Production Selected |
|---|---|---|---|---|---|
| **SVM (Acoustic Features)** | **0.4861** | **0.4822** | **0.3472** | **0.3348** | **YES** |
| **Wav2Vec2 (Fine-Tuned)** | 0.2083 | 0.0984 | 0.1528 | 0.0625 | No (prediction collapse) |

- **Wav2Vec2 Experiment Status**: Evaluated but rejected due to severe prediction collapse (predicting only 2 out of 6 classes on the test set) and poor validation macro-F1 (0.0984).
- **Production Selection**: **SVM** is selected as the final production model based strictly on validation macro-F1.

### Inference Script
Run end-to-end production inference on any `.wav` file using:
```bash
python predict_baseline.py path/to/sample.wav
```
