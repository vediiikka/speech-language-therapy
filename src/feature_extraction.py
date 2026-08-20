import numpy as np
import librosa
from pathlib import Path
from typing import Union

def extract_features(
    audio_path: Union[str, Path],
    sr: int = 16000,
    n_mfcc: int = 20,
    n_fft: int = 2048,
    hop_length: int = 512,
    include_chroma: bool = True,
    include_pitch: bool = True
) -> np.ndarray:
    """
    Fast & High-Performance Acoustic Feature Extraction Pipeline.

    Extracted Acoustic Features:
    - 20 MFCCs (Mel-Frequency Cepstral Coefficients)
    - RMS Energy
    - Zero Crossing Rate (ZCR)
    - Spectral Centroid
    - Spectral Bandwidth
    - Spectral Rolloff
    - Chroma STFT (12 pitch classes)
    - Spectral Contrast & Flatness (Pitch & Tonality proxies)

    Statistical Aggregation (per feature contour):
    - Mean
    - Standard Deviation (std)
    - Minimum (min)
    - Maximum (max)

    Returns:
        Fixed 1D numpy float32 feature array.
    """
    path_str = str(audio_path)

    # Load audio (mono, 16 kHz)
    y, orig_sr = librosa.load(path_str, sr=sr, mono=True)

    if len(y) == 0:
        y = np.zeros(sr, dtype=np.float32)

    # Compute STFT magnitude once for fast feature reuse
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))

    feature_matrices = []

    # 1. MFCCs
    mfccs = librosa.feature.mfcc(S=librosa.amplitude_to_db(S), sr=sr, n_mfcc=n_mfcc)
    feature_matrices.append(mfccs)

    # 2. RMS Energy
    rms = librosa.feature.rms(S=S)
    feature_matrices.append(rms)

    # 3. Zero Crossing Rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=n_fft, hop_length=hop_length)
    feature_matrices.append(zcr)

    # 4. Spectral Centroid
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
    feature_matrices.append(centroid)

    # 5. Spectral Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
    feature_matrices.append(bandwidth)

    # 6. Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)
    feature_matrices.append(rolloff)

    # 7. Chroma STFT (pitch/harmonic contour)
    if include_chroma:
        chroma = librosa.feature.chroma_stft(S=S**2, sr=sr)
        feature_matrices.append(chroma)

    # 8. Spectral Contrast & Flatness (pitch, fundamental timbre & noise ratio)
    if include_pitch:
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        flatness = librosa.feature.spectral_flatness(S=S)
        feature_matrices.append(contrast)
        feature_matrices.append(flatness)

    # Compute 4 summary statistics (mean, std, min, max) for each feature row
    vector_parts = []
    for mat in feature_matrices:
        mean_val = np.mean(mat, axis=1)
        std_val = np.std(mat, axis=1)
        min_val = np.min(mat, axis=1)
        max_val = np.max(mat, axis=1)

        stat_vector = np.concatenate([mean_val, std_val, min_val, max_val])
        vector_parts.append(stat_vector)

    final_feature_vector = np.concatenate(vector_parts).astype(np.float32)
    final_feature_vector = np.nan_to_num(final_feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
    return final_feature_vector
