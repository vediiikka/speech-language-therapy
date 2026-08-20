import sys
import numpy as np
from pathlib import Path

# Add workspace to path
sys.path.append(str(Path(__file__).parent))

from config import BASELINE_ARTIFACTS_DIR, TARGET_CLASSES
from src.utils import calculate_metrics, save_json
from src.models.classical import load_classical_artifacts, get_classical_models


def evaluate_baseline_pipeline():
    print("=" * 60)
    print("STEP 2: CLASSICAL BASELINE EVALUATION (SPEAKER-INDEPENDENT TEST SET)")
    print("=" * 60)

    # 1. Load saved feature matrices
    features_file = BASELINE_ARTIFACTS_DIR / "features.npz"
    if not features_file.exists():
        raise FileNotFoundError(f"Feature matrix not found at {features_file}. Run train_baseline.py first.")

    data = np.load(features_file)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    print(f"Loaded Test set feature matrix: X_test {X_test.shape}, y_test {y_test.shape}")

    # 2. Evaluate saved best model
    best_model, scaler, label_encoder = load_classical_artifacts(BASELINE_ARTIFACTS_DIR)
    
    y_test_pred = best_model.predict(X_test)
    y_test_prob = best_model.predict_proba(X_test)

    best_metrics = calculate_metrics(y_test, y_test_pred, y_test_prob, target_names=TARGET_CLASSES)
    
    print("\n" + "-" * 60)
    print("BEST BASELINE TEST EVALUATION METRICS:")
    print("-" * 60)
    print(f"Accuracy        : {best_metrics['accuracy']:.4f}")
    print(f"Macro Precision : {best_metrics['macro_precision']:.4f}")
    print(f"Macro Recall    : {best_metrics['macro_recall']:.4f}")
    print(f"Macro F1 Score  : {best_metrics['macro_f1']:.4f}")
    print(f"Weighted F1     : {best_metrics['weighted_f1']:.4f}")
    print("\nPer-Class Metrics:")
    for cls_name, p_metrics in best_metrics["per_class"].items():
        print(f"  {cls_name:10s} -> Precision: {p_metrics['precision']:.4f} | Recall: {p_metrics['recall']:.4f} | F1: {p_metrics['f1_score']:.4f} (Support: {p_metrics['support']})")

    # 3. Also evaluate all 3 baseline candidates on Test Set for complete model comparison report
    all_test_metrics = {}
    candidate_models = get_classical_models()
    for name, model in candidate_models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)
        all_test_metrics[name] = calculate_metrics(y_test, pred, prob, target_names=TARGET_CLASSES)

    save_json(best_metrics, BASELINE_ARTIFACTS_DIR / "best_test_metrics.json")
    save_json(all_test_metrics, BASELINE_ARTIFACTS_DIR / "all_baseline_test_metrics.json")

    print("\n" + "=" * 60)
    print("BASELINE CANDIDATE COMPARISON ON TEST SET:")
    print("=" * 60)
    for name, m in all_test_metrics.items():
        print(f"Model: {name:20s} | Acc: {m['accuracy']:.4f} | Macro F1: {m['macro_f1']:.4f} | Weighted F1: {m['weighted_f1']:.4f}")
    print("=" * 60)

    return best_metrics


if __name__ == "__main__":
    evaluate_baseline_pipeline()
