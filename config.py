import os
from pathlib import Path

# ============================================================
# PROJECT & DIRECTORY CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

DATA_RAW_DIR = BASE_DIR / "data" / "raw" / "ravdess"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
BASELINE_ARTIFACTS_DIR = ARTIFACTS_DIR / "baseline"
WAV2VEC2_ARTIFACTS_DIR = ARTIFACTS_DIR / "wav2vec2"

# Ensure directories exist
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
BASELINE_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
WAV2VEC2_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# AUDIO SIGNAL CONFIGURATION
# ============================================================
SAMPLE_RATE = 16000
MONO = True
MAX_DURATION_SEC = 3.5  # Standard fixed length padding/truncation window
NUM_SAMPLES = int(SAMPLE_RATE * MAX_DURATION_SEC)

# ============================================================
# CLASSICAL FEATURE EXTRACTION CONFIGURATION
# ============================================================
FEATURE_CONFIG = {
    "n_mfcc": 20,
    "n_fft": 2048,
    "hop_length": 512,
    "include_mfcc": True,
    "include_rms": True,
    "include_zcr": True,
    "include_spectral_centroid": True,
    "include_spectral_bandwidth": True,
    "include_spectral_rolloff": True,
    "include_chroma": True,
    "include_pitch": True,
    "stats": ["mean", "std", "min", "max"]
}

# ============================================================
# TARGET CLASSES & RAVDESS MAPPING
# ============================================================
# Configurable RAVDESS Emotion Code -> Clinical Affect Class
RAVDESS_EMOTION_MAP = {
    "01": "neutral",   # Neutral
    "02": "neutral",   # Calm -> mapped to neutral
    "03": "happy",     # Happy
    "04": "sad",       # Sad
    "05": "angry",     # Angry
    "06": "anxious",   # Fearful -> mapped to anxious
    "07": "distress",  # Disgust -> mapped to distress
    "08": "happy",     # Surprised -> mapped to happy
}

TARGET_CLASSES = [
    "neutral",
    "happy",
    "sad",
    "angry",
    "anxious",
    "distress"
]

LABEL_TO_ID = {cls_name: i for i, cls_name in enumerate(TARGET_CLASSES)}
ID_TO_LABEL = {i: cls_name for i, cls_name in enumerate(TARGET_CLASSES)}

# ============================================================
# SPEAKER-INDEPENDENT DATA SPLIT CONFIGURATION
# ============================================================
# Total 24 actors: 1-16 Train, 17-20 Validation, 21-24 Test
TRAIN_ACTORS = [f"Actor_{i:02d}" for i in range(1, 17)]
VAL_ACTORS = [f"Actor_{i:02d}" for i in range(17, 21)]
TEST_ACTORS = [f"Actor_{i:02d}" for i in range(21, 25)]

RANDOM_SEED = 42

# ============================================================
# WAV2VEC2 MODEL & TRAINING CONFIGURATION
# ============================================================
MODEL_NAME_OR_PATH = "facebook/wav2vec2-base-960h"
FREEZE_ENCODER = True  # Initially freeze pretrained encoder
DROPOUT_RATE = 0.3
HIDDEN_DIM = 256

# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 3

# Augmentation Settings
ENABLE_AUGMENTATION = True
AUG_NOISE_FACTOR = 0.005
AUG_GAIN_RANGE = (0.8, 1.2)
AUG_TIME_MASK_MAX = 1600  # max 0.1s masked

# Uncertainty threshold for inference
CONFIDENCE_THRESHOLD = 0.45
