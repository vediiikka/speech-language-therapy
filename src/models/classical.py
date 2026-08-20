import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

def get_classical_models(seed=42):
    """
    Instantiate classical ML classifiers with balanced class weighting:
    1. Logistic Regression
    2. Support Vector Machine (SVM)
    3. Random Forest
    """
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=seed
        ),
        "SVM": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=seed
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1
        )
    }
    return models


def save_classical_artifacts(model, scaler, label_encoder, save_dir: Path):
    """Save model, scaler, and label encoder to specified directory."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, save_dir / "best_model.joblib")
    joblib.dump(scaler, save_dir / "scaler.joblib")
    joblib.dump(label_encoder, save_dir / "label_encoder.joblib")


def load_classical_artifacts(save_dir: Path):
    """Load model, scaler, and label encoder from specified directory."""
    save_dir = Path(save_dir)
    model = joblib.load(save_dir / "best_model.joblib")
    scaler = joblib.load(save_dir / "scaler.joblib")
    label_encoder = joblib.load(save_dir / "label_encoder.joblib")
    return model, scaler, label_encoder
