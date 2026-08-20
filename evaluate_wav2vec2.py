import sys
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

# Add workspace to path
sys.path.append(str(Path(__file__).parent))

from config import (
    DATA_RAW_DIR,
    WAV2VEC2_ARTIFACTS_DIR,
    BASELINE_ARTIFACTS_DIR,
    TARGET_CLASSES,
    BATCH_SIZE
)
from src.utils import calculate_metrics, save_json, load_json
from src.dataset import load_dataset_file_records, get_speaker_independent_splits, SpeechEmotionDataset
from src.models.wav2vec2 import Wav2Vec2SpeechClassifier


def evaluate_wav2vec2_pipeline():
    print("=" * 60)
    print("STEP 4: WAV2VEC2 EVALUATION & CROSS-MODEL COMPARISON")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load Test split (Actors 21-24)
    records = load_dataset_file_records(DATA_RAW_DIR)
    _, _, test_records = get_speaker_independent_splits(records)
    print(f"Evaluating Wav2Vec2 on Test Set ({len(test_records)} records)...")

    test_dataset = SpeechEmotionDataset(test_records, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Load Model Checkpoint & Config
    config_file = WAV2VEC2_ARTIFACTS_DIR / "config.json"
    weights_file = WAV2VEC2_ARTIFACTS_DIR / "best_model.pt"

    if not config_file.exists() or not weights_file.exists():
        raise FileNotFoundError(f"Wav2Vec2 model artifacts not found at {WAV2VEC2_ARTIFACTS_DIR}. Run train_wav2vec2.py first.")

    model_config = load_json(config_file)
    model = Wav2Vec2SpeechClassifier(
        model_name_or_path=model_config["model_name_or_path"],
        num_classes=model_config["num_classes"],
        hidden_dim=model_config["hidden_dim"],
        dropout_rate=model_config["dropout_rate"],
        freeze_encoder=model_config["freeze_encoder"]
    ).to(device)

    model.load_state_dict(torch.load(weights_file, map_location=device))
    model.eval()

    # 3. Perform Test Inference
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["input_values"].to(device)
            labels = batch["labels"].to(device)

            logits = model(inputs)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    w2v_metrics = calculate_metrics(
        np.array(all_targets),
        np.array(all_preds),
        np.array(all_probs),
        target_names=TARGET_CLASSES
    )

    save_json(w2v_metrics, WAV2VEC2_ARTIFACTS_DIR / "test_metrics.json")

    print("\n" + "-" * 60)
    print("WAV2VEC2 MODEL TEST METRICS:")
    print("-" * 60)
    print(f"Accuracy        : {w2v_metrics['accuracy']:.4f}")
    print(f"Macro Precision : {w2v_metrics['macro_precision']:.4f}")
    print(f"Macro Recall    : {w2v_metrics['macro_recall']:.4f}")
    print(f"Macro F1 Score  : {w2v_metrics['macro_f1']:.4f}")
    print(f"Weighted F1     : {w2v_metrics['weighted_f1']:.4f}")
    print("\nPer-Class Metrics:")
    for cls_name, p_metrics in w2v_metrics["per_class"].items():
        print(f"  {cls_name:10s} -> Precision: {p_metrics['precision']:.4f} | Recall: {p_metrics['recall']:.4f} | F1: {p_metrics['f1_score']:.4f} (Support: {p_metrics['support']})")

    # 4. Comprehensive Cross-Model Comparison Report (Baseline vs. Wav2Vec2)
    print("\n" + "=" * 80)
    print("ALL MODELS COMPARISON REPORT (SPEAKER-INDEPENDENT TEST SET)")
    print("=" * 80)
    print(f"{'Model':25s} | {'Accuracy':10s} | {'Macro Precision':16s} | {'Macro Recall':14s} | {'Macro F1':10s} | {'Weighted F1':12s}")
    print("-" * 85)

    comparison = {}

    # Load baseline candidate test metrics if available
    baseline_all_file = BASELINE_ARTIFACTS_DIR / "all_baseline_test_metrics.json"
    if baseline_all_file.exists():
        base_models = load_json(baseline_all_file)
        for name, m in base_models.items():
            comparison[name] = m
            print(f"{name:25s} | {m['accuracy']:10.4f} | {m['macro_precision']:16.4f} | {m['macro_recall']:14.4f} | {m['macro_f1']:10.4f} | {m['weighted_f1']:12.4f}")

    comparison["Wav2Vec2"] = w2v_metrics
    print(f"{'Wav2Vec2 (Deep Affect)':25s} | {w2v_metrics['accuracy']:10.4f} | {w2v_metrics['macro_precision']:16.4f} | {w2v_metrics['macro_recall']:14.4f} | {w2v_metrics['macro_f1']:10.4f} | {w2v_metrics['weighted_f1']:12.4f}")
    print("=" * 85)

    # Determine Overall Champion Model based on Macro F1
    champion_name = max(comparison, key=lambda k: comparison[k]["macro_f1"])
    print(f"\n[*] OVERALL CHAMPION MODEL: {champion_name} (Test Macro F1: {comparison[champion_name]['macro_f1']:.4f})")
    print("=" * 80)

    summary_report = {
        "models_compared": comparison,
        "champion_model": champion_name,
        "champion_macro_f1": comparison[champion_name]["macro_f1"]
    }
    save_json(summary_report, WAV2VEC2_ARTIFACTS_DIR / "model_comparison_report.json")

    return w2v_metrics


if __name__ == "__main__":
    evaluate_wav2vec2_pipeline()
