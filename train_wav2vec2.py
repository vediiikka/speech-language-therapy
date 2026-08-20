import sys
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add workspace to path
sys.path.append(str(Path(__file__).parent))

from config import (
    DATA_RAW_DIR,
    WAV2VEC2_ARTIFACTS_DIR,
    TARGET_CLASSES,
    LABEL_TO_ID,
    ID_TO_LABEL,
    MODEL_NAME_OR_PATH,
    FREEZE_ENCODER,
    DROPOUT_RATE,
    HIDDEN_DIM,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    RANDOM_SEED
)
from src.utils import set_seed, calculate_metrics, save_json
from src.dataset import load_dataset_file_records, get_speaker_independent_splits, SpeechEmotionDataset
from src.models.wav2vec2 import Wav2Vec2SpeechClassifier


def train_wav2vec2_pipeline():
    print("=" * 60)
    print("STEP 3: WAV2VEC2 ADVANCED MODEL TRAINING")
    print("=" * 60)

    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Device: {device}")

    # 1. Load dataset records & speaker-independent splits
    records = load_dataset_file_records(DATA_RAW_DIR)
    train_records, val_records, test_records = get_speaker_independent_splits(records)

    print(f"Train samples : {len(train_records)}")
    print(f"Val samples   : {len(val_records)}")
    print(f"Test samples  : {len(test_records)}")

    # 2. Compute class-imbalance weights
    train_labels = [r["label_id"] for r in train_records]
    class_counts = np.bincount(train_labels, minlength=len(TARGET_CLASSES))
    print("\nTraining Class Distribution:")
    for idx, name in enumerate(TARGET_CLASSES):
        print(f"  {name:10s}: {class_counts[idx]} samples")

    # Inverse frequency class weighting
    total_samples = len(train_labels)
    class_weights = total_samples / (len(TARGET_CLASSES) * np.maximum(class_counts, 1).astype(np.float32))
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # 3. Create Datasets and DataLoaders
    train_dataset = SpeechEmotionDataset(train_records, is_train=True)
    val_dataset = SpeechEmotionDataset(val_records, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Instantiate Model
    print(f"\nInitializing Wav2Vec2 Model (Pretrained: {MODEL_NAME_OR_PATH}, Frozen Encoder: {FREEZE_ENCODER})...")
    model = Wav2Vec2SpeechClassifier(
        model_name_or_path=MODEL_NAME_OR_PATH,
        num_classes=len(TARGET_CLASSES),
        hidden_dim=HIDDEN_DIM,
        dropout_rate=DROPOUT_RATE,
        freeze_encoder=FREEZE_ENCODER
    ).to(device)

    # 5. Loss, Optimizer & Scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    # 6. Training & Validation Loop
    best_val_f1 = -1.0
    patience_counter = 0
    history = []

    print("\nStarting Wav2Vec2 Affect Training Loop...")
    print("-" * 60)

    for epoch in range(1, NUM_EPOCHS + 1):
        # TRAIN EPOCH
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs = batch["input_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss = train_loss / len(train_dataset)

        # VALIDATION EPOCH
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input_values"].to(device)
                labels = batch["labels"].to(device)

                logits = model(inputs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * inputs.size(0)

                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        val_loss = val_loss / len(val_dataset)
        val_metrics = calculate_metrics(np.array(all_targets), np.array(all_preds), np.array(all_probs), target_names=TARGET_CLASSES)
        val_macro_f1 = val_metrics["macro_f1"]
        val_acc = val_metrics["accuracy"]

        scheduler.step(val_macro_f1)

        epoch_record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc),
            "val_macro_f1": float(val_macro_f1),
            "val_weighted_f1": float(val_metrics["weighted_f1"])
        }
        history.append(epoch_record)

        print(
            f"Epoch [{epoch:02d}/{NUM_EPOCHS:02d}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Macro F1: {val_macro_f1:.4f}"
        )

        # Checkpoint Best Model
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            patience_counter = 0
            torch.save(model.state_dict(), WAV2VEC2_ARTIFACTS_DIR / "best_model.pt")
            save_json(val_metrics, WAV2VEC2_ARTIFACTS_DIR / "best_val_metrics.json")
            print(f"  [*] New best model checkpoint saved! (Val Macro F1: {val_macro_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch} epochs without improvement.")
                break

    print("-" * 60)
    print(f"Wav2Vec2 Training Complete. Best Validation Macro F1: {best_val_f1:.4f}")

    # 7. Save Artifacts & Configurations
    model_config = {
        "model_name_or_path": MODEL_NAME_OR_PATH,
        "num_classes": len(TARGET_CLASSES),
        "hidden_dim": HIDDEN_DIM,
        "dropout_rate": DROPOUT_RATE,
        "freeze_encoder": FREEZE_ENCODER,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "model_version": "1.0.0-wav2vec2"
    }

    label_mapping = {
        "label_to_id": LABEL_TO_ID,
        "id_to_label": ID_TO_LABEL,
        "target_classes": TARGET_CLASSES
    }

    save_json(model_config, WAV2VEC2_ARTIFACTS_DIR / "config.json")
    save_json(label_mapping, WAV2VEC2_ARTIFACTS_DIR / "label_mapping.json")
    save_json(history, WAV2VEC2_ARTIFACTS_DIR / "training_history.json")

    print(f"Wav2Vec2 model artifacts saved to: {WAV2VEC2_ARTIFACTS_DIR}")
    return best_val_f1


if __name__ == "__main__":
    train_wav2vec2_pipeline()
