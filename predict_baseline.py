import sys
import numpy as np
import json
from pathlib import Path
from typing import Union, Dict, Any

# Add workspace to path
sys.path.append(str(Path(__file__).parent))

from config import BASELINE_ARTIFACTS_DIR, CONFIDENCE_THRESHOLD, TARGET_CLASSES, DATA_RAW_DIR
from src.feature_extraction import extract_features
from src.models.classical import load_classical_artifacts


class BaselinePredictor:
    """
    Inference Engine for Classical ML Affect Classifier.
    """
    def __init__(self, artifacts_dir: Path = BASELINE_ARTIFACTS_DIR, threshold: float = CONFIDENCE_THRESHOLD):
        self.artifacts_dir = Path(artifacts_dir)
        self.threshold = threshold
        try:
            self.model, self.scaler, self.label_encoder = load_classical_artifacts(self.artifacts_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to load production model artifacts: {str(e)}")

    def predict(self, audio_path: Union[str, Path]) -> Dict[str, Any]:
        path = Path(audio_path)
        
        # 1. Validate file existence
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        # 2. Validate file extension
        if path.suffix.lower() != ".wav":
            raise ValueError(f"Unsupported file format '{path.suffix}'. Only standard PCM .wav files are supported.")

        # 3. Extract 1D fixed acoustic features with error handling for corrupt files
        try:
            features = extract_features(path)
        except Exception as e:
            raise ValueError(f"Failed to parse or process audio file. It may be corrupt or invalid: {str(e)}")

        # 4. Scale features
        try:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
        except Exception as e:
            raise ValueError(f"Feature scaling mismatch: {str(e)}")

        # 5. Model Inference (Probabilities)
        try:
            probs = self.model.predict_proba(features_scaled)[0]
        except Exception as e:
            raise ValueError(f"Model prediction failed: {str(e)}")

        # Map to class dictionary
        prob_dict = {
            cls_name: float(round(probs[idx], 4))
            for idx, cls_name in enumerate(self.label_encoder.classes_)
        }

        # Determine dominant state & model confidence
        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])
        top_class = str(self.label_encoder.classes_[top_idx])

        # Clinical uncertainty / confidence status division
        if top_prob < self.threshold:
            confidence_status = "Low Confidence"
            # Return 'unknown' as affect state when uncertainty is too high
            dominant_state = "unknown"
        else:
            confidence_status = "Confident"
            dominant_state = top_class

        return {
            "status": "success",
            "file_processed": str(path.resolve()),
            "model_used": type(self.model).__name__,
            "clinical_affect_classification": dominant_state,
            "confidence_score": float(round(top_prob, 4)),
            "confidence_status": confidence_status,
            "class_probabilities": prob_dict,
            "disclaimer": "This output is a speech emotion classification for clinical affect tracking. It is NOT a clinical diagnosis."
        }


def predict_baseline(audio_path: Union[str, Path], threshold: float = CONFIDENCE_THRESHOLD) -> Dict[str, Any]:
    predictor = BaselinePredictor(threshold=threshold)
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

    try:
        result = predict_baseline(test_file)
        print(json.dumps(result, indent=4))
    except Exception as e:
        error_result = {
            "status": "error",
            "error_message": str(e),
            "disclaimer": "This output is a speech emotion classification for clinical affect tracking. It is NOT a clinical diagnosis."
        }
        print(json.dumps(error_result, indent=4))
        sys.exit(1)
