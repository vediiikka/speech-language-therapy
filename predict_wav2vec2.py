import sys
import torch
import librosa
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any

# Add workspace to path
sys.path.append(str(Path(__file__).parent))

from config import (
    WAV2VEC2_ARTIFACTS_DIR,
    CONFIDENCE_THRESHOLD,
    SAMPLE_RATE,
    NUM_SAMPLES,
    TARGET_CLASSES,
    ID_TO_LABEL,
    DATA_RAW_DIR
)
from src.utils import load_json
from src.models.wav2vec2 import Wav2Vec2SpeechClassifier


class Wav2Vec2Predictor:
    """
    Inference Engine for Wav2Vec2 Speech Affect Classifier.
    """
    def __init__(self, artifacts_dir: Path = WAV2VEC2_ARTIFACTS_DIR, threshold: float = CONFIDENCE_THRESHOLD):
        self.artifacts_dir = Path(artifacts_dir)
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load configurations and weights
        config_path = self.artifacts_dir / "config.json"
        weights_path = self.artifacts_dir / "best_model.pt"

        if not config_path.exists() or not weights_path.exists():
            raise FileNotFoundError(f"Wav2Vec2 artifacts missing in {self.artifacts_dir}. Run train_wav2vec2.py first.")

        self.config = load_json(config_path)
        self.label_mapping = load_json(self.artifacts_dir / "label_mapping.json")

        self.model = Wav2Vec2SpeechClassifier(
            model_name_or_path=self.config["model_name_or_path"],
            num_classes=self.config["num_classes"],
            hidden_dim=self.config["hidden_dim"],
            dropout_rate=self.config["dropout_rate"],
            freeze_encoder=self.config["freeze_encoder"]
        ).to(self.device)

        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

    def predict(self, audio_path: Union[str, Path]) -> Dict[str, Any]:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        # 1. Load raw audio (mono, 16 kHz)
        y, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)

        # 2. Peak normalization
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

        # 3. Pad or truncate to standard NUM_SAMPLES window
        if len(y) < NUM_SAMPLES:
            pad_len = NUM_SAMPLES - len(y)
            y = np.pad(y, (0, pad_len), mode="constant")
        else:
            y = y[:NUM_SAMPLES]

        input_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(self.device)

        # 4. Forward Inference Pass
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        # 5. Build Probabilities Dictionary
        prob_dict = {}
        for idx, cls_name in enumerate(TARGET_CLASSES):
            prob_dict[cls_name] = float(round(probs[idx], 4))

        # 6. Top class and confidence thresholding
        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])
        top_class = TARGET_CLASSES[top_idx]

        if top_prob < self.threshold:
            dominant_state = "unknown"
            uncertain = True
        else:
            dominant_state = top_class
            uncertain = False

        return {
            "dominant_state": dominant_state,
            "confidence": float(round(top_prob, 4)),
            "probabilities": prob_dict,
            "uncertain": uncertain
        }


def predict_wav2vec2(audio_path: Union[str, Path], threshold: float = CONFIDENCE_THRESHOLD) -> Dict[str, Any]:
    predictor = Wav2Vec2Predictor(threshold=threshold)
    return predictor.predict(audio_path)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        # Default test sample if no file argument supplied
        test_file = DATA_RAW_DIR / "Actor_01" / "03-01-01-01-01-01-01.wav"
        if not test_file.exists():
            wav_candidates = list(DATA_RAW_DIR.glob("**/*.wav"))
            test_file = wav_candidates[0] if wav_candidates else "sample.wav"

    print(f"Running Wav2Vec2 Model Inference on: {test_file}")
    result = predict_wav2vec2(test_file)
    import json
    print(json.dumps(result, indent=4))
