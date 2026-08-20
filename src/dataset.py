import os
import random
import torch
import librosa
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from torch.utils.data import Dataset

from config import (
    DATA_RAW_DIR,
    RAVDESS_EMOTION_MAP,
    TARGET_CLASSES,
    LABEL_TO_ID,
    TRAIN_ACTORS,
    VAL_ACTORS,
    TEST_ACTORS,
    SAMPLE_RATE,
    NUM_SAMPLES,
    ENABLE_AUGMENTATION,
    AUG_NOISE_FACTOR,
    AUG_GAIN_RANGE,
    AUG_TIME_MASK_MAX
)


def load_dataset_file_records(data_dir: Path = DATA_RAW_DIR) -> List[Dict]:
    """
    Scan dataset directory and return records with file path, actor, and mapped emotion label.
    """
    records = []
    data_dir = Path(data_dir)
    
    for actor_dir in sorted(data_dir.glob("Actor_*")):
        actor_name = actor_dir.name
        for wav_file in sorted(actor_dir.glob("*.wav")):
            parts = wav_file.stem.split("-")
            if len(parts) == 7:
                emotion_code = parts[2]
                if emotion_code in RAVDESS_EMOTION_MAP:
                    mapped_emotion = RAVDESS_EMOTION_MAP[emotion_code]
                    records.append({
                        "path": str(wav_file),
                        "filename": wav_file.name,
                        "actor": actor_name,
                        "emotion": mapped_emotion,
                        "label_id": LABEL_TO_ID[mapped_emotion]
                    })
    return records


def get_speaker_independent_splits(records: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Partition dataset into Train, Validation, and Test sets based on strict speaker independence.
    """
    train_records = [r for r in records if r["actor"] in TRAIN_ACTORS]
    val_records = [r for r in records if r["actor"] in VAL_ACTORS]
    test_records = [r for r in records if r["actor"] in TEST_ACTORS]
    return train_records, val_records, test_records


# ============================================================
# SPEECH DATA AUGMENTATION (SECTION 6)
# ============================================================

def apply_audio_augmentation(waveform: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Realistic affective speech augmentation:
    1. Small additive white Gaussian noise (SNR preserving label)
    2. Gain / volume scaling
    3. Time masking (short zero mask frame)
    """
    aug_wave = waveform.copy()

    # 1. Gain variation
    gain = random.uniform(*AUG_GAIN_RANGE)
    aug_wave = aug_wave * gain

    # 2. Small background noise
    if AUG_NOISE_FACTOR > 0:
        noise = np.random.randn(len(aug_wave)) * AUG_NOISE_FACTOR
        aug_wave = aug_wave + noise

    # 3. Time masking
    if AUG_TIME_MASK_MAX > 0 and len(aug_wave) > AUG_TIME_MASK_MAX:
        mask_len = random.randint(100, AUG_TIME_MASK_MAX)
        mask_start = random.randint(0, len(aug_wave) - mask_len)
        aug_wave[mask_start : mask_start + mask_len] = 0.0

    # Ensure peak amplitude normalization
    max_val = np.max(np.abs(aug_wave))
    if max_val > 0:
        aug_wave = aug_wave / max_val

    return aug_wave.astype(np.float32)


# ============================================================
# PYTORCH DATASET FOR RAW WAVEFORMS (SECTION 3)
# ============================================================

class SpeechEmotionDataset(Dataset):
    """
    PyTorch Dataset for Raw Audio Waveforms fed into Wav2Vec2.
    Processes audio to:
    - Mono
    - 16 kHz sample rate
    - Fixed length padding / truncation
    - Normalized float32 waveform [-1.0, 1.0]
    """
    def __init__(self, records: List[Dict], is_train: bool = False, max_samples: int = NUM_SAMPLES):
        self.records = records
        self.is_train = is_train
        self.max_samples = max_samples

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        file_path = record["path"]
        label_id = record["label_id"]

        # Load raw audio (mono, 16 kHz)
        y, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

        # Apply augmentation if in training mode
        if self.is_train and ENABLE_AUGMENTATION:
            y = apply_audio_augmentation(y)
        else:
            # Peak normalize
            max_val = np.max(np.abs(y))
            if max_val > 0:
                y = y / max_val

        # Zero-mean unit-variance scaling for Wav2Vec2 compatibility
        y_mean = y.mean()
        y_std = y.std()
        if y_std > 0:
            y = (y - y_mean) / (y_std + 1e-7)
        else:
            y = y - y_mean

        # Pad or truncate to max_samples
        if len(y) < self.max_samples:
            pad_len = self.max_samples - len(y)
            y = np.pad(y, (0, pad_len), mode="constant")
        else:
            y = y[:self.max_samples]

        return {
            "input_values": torch.tensor(y, dtype=torch.float32),
            "labels": torch.tensor(label_id, dtype=torch.long),
            "file_path": file_path,
            "actor": record["actor"]
        }
