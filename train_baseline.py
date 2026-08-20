import sys
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Add workspace to path
sys.path.append(str(Path(__file__).parent))

from config import (
    DATA_RAW_DIR,
    BASELINE_ARTIFACTS_DIR,
    TARGET_CLASSES,
    FEATURE_CONFIG,
    RANDOM_SEED
)
from src.utils import set_seed, calculate_metrics, save_json
from src.feature_extraction import extract_features
from src.dataset import load_dataset_file_records, get_speaker_independent_splits
from src.models.classical import get_classical_models, save_classical_artifacts


def train_baseline_pipeline():
    print("=" * 60)
    print("STEP 1: CLASSICAL ML BASELINE TRAINING & SELECTION")
    print("=" * 60)

    set_seed(RANDOM_SEED)

    # 1. Load dataset file records
    records = load_dataset_file_records(DATA_RAW_DIR)
    print(f"Total audio files found: {len(records)}")

    # 2. Get speaker-independent train / validation / test splits
    train_records, val_records, test_records = get_speaker_independent_splits(records)
    print(f"Train split records : {len(train_records)} (Actors 01-16)")
    print(f"Val split records   : {len(val_records)} (Actors 17-20)")
    print(f"Test split records  : {len(test_records)} (Actors 21-24)")

    # 3. Extract fixed 1D acoustic features
    def extract_set_features(record_list, set_name):
        print(f"\nExtracting Librosa acoustic features for {set_name}...")
        X_list, y_list = [], []
        for idx, r in enumerate(record_list):
            feat = extract_features(r["path"])
            X_list.append(feat)
            y_list.append(r["emotion"])
            if (idx + 1) % 50 == 0 or (idx + 1) == len(record_list):
                print(f"  [{set_name}] Processed {idx + 1}/{len(record_list)} files...")
        return np.array(X_list), np.array(y_list)

    X_train, y_train_str = extract_set_features(train_records, "TRAIN")
    X_val, y_val_str = extract_set_features(val_records, "VAL")
    X_test, y_test_str = extract_set_features(test_records, "TEST")

    print(f"\nFeature Matrix Dimensions:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val  : {X_val.shape}")
    print(f"  X_test : {X_test.shape}")

    # 4. Standardize features (fit on train set only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(TARGET_CLASSES)
    y_train = label_encoder.transform(y_train_str)
    y_val = label_encoder.transform(y_val_str)
    y_test = label_encoder.transform(y_test_str)

    # Save feature & split matrices for reproducible downstream evaluation
    np.savez_compressed(
        BASELINE_ARTIFACTS_DIR / "features.npz",
        X_train=X_train_scaled, y_train=y_train,
        X_val=X_val_scaled, y_val=y_val,
        X_test=X_test_scaled, y_test=y_test
    )

    # 5. Train & Evaluate Candidates on Validation Set for Model Selection
    models = get_classical_models(seed=RANDOM_SEED)
    best_model_name = None
    best_val_f1 = -1.0
    best_model_obj = None
    val_summary = {}

    print("\nTraining and evaluating candidate baseline classifiers...")
    print("-" * 60)

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_val_pred = model.predict(X_val_scaled)
        y_val_prob = model.predict_proba(X_val_scaled)

        metrics = calculate_metrics(y_val, y_val_pred, y_val_prob, target_names=TARGET_CLASSES)
        val_f1 = metrics["macro_f1"]
        val_summary[name] = metrics

        print(f"Model: {name:20s} | Val Acc: {metrics['accuracy']:.4f} | Val Macro F1: {val_f1:.4f} | Val Weighted F1: {metrics['weighted_f1']:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_name = name
            best_model_obj = model

    print("-" * 60)
    print(f"[*] Selected Best Baseline Model: {best_model_name} (Val Macro F1: {best_val_f1:.4f})")

    # 6. Save Artifacts
    save_classical_artifacts(best_model_obj, scaler, label_encoder, BASELINE_ARTIFACTS_DIR)

    feature_meta = {
        "feature_config": FEATURE_CONFIG,
        "feature_dim": int(X_train.shape[1]),
        "target_classes": TARGET_CLASSES
    }
    save_json(feature_meta, BASELINE_ARTIFACTS_DIR / "feature_config.json")

    model_meta = {
        "selected_model": best_model_name,
        "all_candidates": list(models.keys()),
        "model_version": "1.0.0-baseline"
    }
    save_json(model_meta, BASELINE_ARTIFACTS_DIR / "model_config.json")
    save_json(val_summary, BASELINE_ARTIFACTS_DIR / "val_metrics.json")

    print(f"Baseline training artifacts saved to: {BASELINE_ARTIFACTS_DIR}")
    return best_model_name, best_val_f1


if __name__ == "__main__":
    train_baseline_pipeline()
