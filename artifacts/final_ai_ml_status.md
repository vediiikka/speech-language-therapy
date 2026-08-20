# Final AI/ML Status Report

This document reports the final pipeline status, evaluation results, and production decisions for the Speech Emotion Recognition (SER) / Clinical Affect Detection pipeline.

## 1. Dataset
The dataset consists of **432 raw audio WAV files** from the RAVDESS corpus. Each file is verified to have a unique SHA-256 hash. The files are distributed across 24 actors (Actor_01 through Actor_24) with 18 audio samples per actor. 

The targets are mapped into six clinical affect classes:
- `neutral`
- `happy`
- `sad`
- `angry`
- `anxious`
- `distress`

The dataset is perfectly balanced, containing exactly 72 samples per class across the entire corpus.

## 2. Speaker-Independent Split
To ensure unbiased evaluation and prevent data leakage, the split is strictly partitioned by actor identity:
- **Train Set**: Actor_01 to Actor_16 (288 files, 48 per class)
- **Validation Set**: Actor_17 to Actor_20 (72 files, 12 per class)
- **Test Set**: Actor_21 to Actor_24 (72 files, 12 per class)

## 3. Baseline Feature Extraction
For the classical models, 1D acoustic statistical features are extracted from the raw waveforms:
- **Features Extracted**: MFCCs, Spectral Centroid, Spectral Bandwidth, Spectral Rolloff, Root Mean Square (RMS) energy, Zero Crossing Rate (ZCR), Chroma STFT, and Pitch.
- **Statistical Aggregation**: Mean, standard deviation, minimum, and maximum are computed across frames, yielding a **180-dimensional feature vector** per file.
- **Scaling**: Standard scaling is fit only on the train set and saved to `artifacts/baseline/scaler.joblib`.

## 4. SVM Baseline
The baseline candidates (Logistic Regression, SVM, and Random Forest) were trained using class-balanced weights:
- SVM validation macro-F1: **0.4822**
- SVM validation accuracy: **0.4861**
- The SVM model was saved to `artifacts/baseline/best_model.joblib`.

## 5. Wav2Vec2 Experiment
An advanced deep learning classifier based on `facebook/wav2vec2-base-960h` was experimentally trained.
- **First Run**: Unnormalized waveforms caused input-distribution mismatch, leading to complete prediction collapse.
- **Correction**: Preprocessing was fixed to include standard zero-mean unit-variance scaling. 
- **Retraining**: Corrected Wav2Vec2 retraining was performed using an embedding-cache strategy on the frozen backbone. However, Wav2Vec2 still suffered from severe prediction collapse, mapping 94% of test inputs to `angry` and the rest to `happy`.
- **Wav2Vec2 was rejected as the production model** due to this collapse and its poor macro-F1.

## 6. Evaluation Results
The final metrics across validation and held-out test splits are:

| Model | Val Accuracy | Val Macro-F1 | Test Accuracy | Test Macro-F1 | Classes Predicted (Test) |
|---|---|---|---|---|---|
| **SVM (Baseline)** | **0.4861** | **0.4822** | **0.3472** | **0.3348** | **6/6** |
| **Wav2Vec2 (Retrained)** | 0.2083 | 0.0984 | 0.1528 | 0.0625 | 2/6 |

## 7. Final Model Selection
- **Selected Model**: **SVM**
- **Selection Criterion**: Validation macro-F1 performance only. The test set was held-out strictly as a final unbiased evaluation set.
- **Logistic Regression Rejection**: The old comparison report's recommendation of Logistic Regression was based on test-set macro-F1. Since test performance must not guide model selection, that selection was methodologically invalid and rejected.

## 8. Production Inference
The production engine [`predict_baseline.py`](file:///c:/Users/Dell/Downloads/ml_shareable/ml/predict_baseline.py) runs end-to-end inference:
1. Validates the raw `.wav` file structure.
2. Extracts the 180-dimensional acoustic features.
3. Scales features using the saved scaler.
4. Performs probability estimation using the saved SVM model.
5. Divides results based on a confidence threshold (Confident vs. Low Confidence).
6. Formats structured clinical affect classifications with a warning disclaimer that outputs are NOT diagnostic.

## 9. Limitations
- **Small Dataset**: 432 samples limit deep learning models (like Wav2Vec2) from fine-tuning classification headers effectively without overfitting/collapsing.
- **Out of Domain Performance**: RAVDESS consists of clean actor speech; real-world noisy clinical environments might see degraded performance.

## 10. Future Improvements
- **Data Augmentation expansion**: Incorporate synthetic noise models and vocal tract length perturbation (VTLP).
- **Unfrozen Fine-tuning**: Train the Wav2Vec2 encoder directly on a larger external speech emotion corpus before clinical transfer learning.
