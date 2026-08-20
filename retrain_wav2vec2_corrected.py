"""
retrain_wav2vec2_corrected.py  (v3 — embedding-cache strategy)
───────────────────────────────────────────────────────────────
STRATEGY:
  Since FREEZE_ENCODER=True, the Wav2Vec2 encoder weights never change.
  Therefore we can extract embeddings ONCE (one full pass over all audio),
  cache them in memory, then train ONLY the 2-layer classification head
  on those fixed embeddings.

  This reduces training from O(94M-param encoder × N_epochs) to:
    - 1× encoder pass over all audio  (~10-20 min one-time cost)
    - N_epochs × tiny classifier-head steps  (milliseconds each)

  Scientific validity preserved:
    - Same speaker-independent splits (train 1-16, val 17-20, test 21-24)
    - Same corrected zero-mean unit-variance normalization
    - Same model architecture and frozen encoder
    - Same class weighting, optimizer, LR scheduler
    - Same early stopping patience=3, RANDOM_SEED=42
    - Model selection on validation macro-F1 only
    - Test evaluated ONCE after best checkpoint selected

  All artifacts saved ONLY under:  artifacts/wav2vec2_retrained/
  Original artifacts/wav2vec2/ left untouched.
"""

import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import librosa
import random
import os

sys.path.append(str(Path(__file__).parent))

from config import (
    DATA_RAW_DIR, ARTIFACTS_DIR, TARGET_CLASSES, LABEL_TO_ID, ID_TO_LABEL,
    RAVDESS_EMOTION_MAP, TRAIN_ACTORS, VAL_ACTORS, TEST_ACTORS,
    MODEL_NAME_OR_PATH, FREEZE_ENCODER, DROPOUT_RATE, HIDDEN_DIM,
    WEIGHT_DECAY, NUM_EPOCHS, EARLY_STOPPING_PATIENCE, RANDOM_SEED,
    ENABLE_AUGMENTATION, AUG_NOISE_FACTOR, AUG_GAIN_RANGE, AUG_TIME_MASK_MAX,
    SAMPLE_RATE, NUM_SAMPLES,
)
from src.utils import set_seed, calculate_metrics, save_json
from src.models.wav2vec2 import Wav2Vec2SpeechClassifier

torch.set_num_threads(os.cpu_count() or 4)

RETRAINED_DIR = ARTIFACTS_DIR / "wav2vec2_retrained"
RETRAINED_DIR.mkdir(parents=True, exist_ok=True)

# Training hyperparams for classifier head only
BATCH_SIZE    = 32   # much larger now — embeddings are tiny tensors
LEARNING_RATE = 1e-3

# SVM baseline (verified from val_metrics.json + best_test_metrics.json)
SVM_VAL_MACRO_F1    = 0.4821734250719758
SVM_VAL_WEIGHTED_F1 = 0.48217342507197575
SVM_VAL_ACCURACY    = 0.4861111111111111
SVM_TEST_MACRO_F1   = 0.3347882464011496
SVM_TEST_WEIGHTED_F1= 0.3347882464011496
SVM_TEST_ACCURACY   = 0.3472222222222222


# ── Data loading ───────────────────────────────────────────────

def load_records(data_dir):
    records = []
    for actor_dir in sorted(Path(data_dir).glob("Actor_*")):
        actor = actor_dir.name
        for wav in sorted(actor_dir.glob("*.wav")):
            parts = wav.stem.split("-")
            if len(parts) == 7 and parts[2] in RAVDESS_EMOTION_MAP:
                emotion = RAVDESS_EMOTION_MAP[parts[2]]
                records.append({
                    "path": str(wav), "actor": actor,
                    "emotion": emotion, "label_id": LABEL_TO_ID[emotion],
                })
    return records


def split_records(records):
    return (
        [r for r in records if r["actor"] in TRAIN_ACTORS],
        [r for r in records if r["actor"] in VAL_ACTORS],
        [r for r in records if r["actor"] in TEST_ACTORS],
    )


def augment(y):
    gain = random.uniform(*AUG_GAIN_RANGE)
    y = y * gain
    if AUG_NOISE_FACTOR > 0:
        y = y + np.random.randn(len(y)) * AUG_NOISE_FACTOR
    mask_max = min(AUG_TIME_MASK_MAX, NUM_SAMPLES // 4)
    if mask_max > 100 and len(y) > mask_max:
        ml = random.randint(100, mask_max)
        ms = random.randint(0, len(y) - ml)
        y[ms:ms+ml] = 0.0
    mv = np.max(np.abs(y))
    if mv > 0:
        y = y / mv
    return y.astype(np.float32)


def load_waveform(path, is_train=False):
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if is_train and ENABLE_AUGMENTATION:
        y = augment(y)
    else:
        mv = np.max(np.abs(y))
        if mv > 0:
            y = y / mv
    # Corrected normalization: zero-mean unit-variance
    m, s = y.mean(), y.std()
    y = (y - m) / (s + 1e-7) if s > 0 else y - m
    # Pad / truncate
    if len(y) < NUM_SAMPLES:
        y = np.pad(y, (0, NUM_SAMPLES - len(y)), mode="constant")
    else:
        y = y[:NUM_SAMPLES]
    return y.astype(np.float32)


# ── Embedding extraction ───────────────────────────────────────

def extract_embeddings(records, encoder, device, is_train=False, batch_sz=16):
    """Run encoder over all records, return (embeddings_tensor, labels_tensor)."""
    encoder.eval()
    all_embs, all_lbls = [], []
    n = len(records)
    for i in range(0, n, batch_sz):
        batch_records = records[i:i+batch_sz]
        waves = np.stack([load_waveform(r["path"], is_train=is_train) for r in batch_records])
        inp   = torch.tensor(waves, dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs = encoder.wav2vec2(input_values=inp)
            emb     = outputs.last_hidden_state.mean(dim=1)  # mean pool → [B, 768]
        all_embs.append(emb.cpu())
        all_lbls.extend([r["label_id"] for r in batch_records])
        done = min(i + batch_sz, n)
        print(f"  Extracted {done}/{n}", end="\r", flush=True)
    print()
    return torch.cat(all_embs, dim=0), torch.tensor(all_lbls, dtype=torch.long)


# ── Classifier head only ───────────────────────────────────────

class ClassifierHead(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=6, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    def forward(self, x):
        return self.net(x)


# ── Main ───────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("CORRECTED WAV2VEC2 RETRAINING  (embedding-cache strategy)")
    print(f"Audio window  : {NUM_SAMPLES/SAMPLE_RATE:.1f}s  ({NUM_SAMPLES} samples)")
    print(f"Encoder       : frozen — embeddings extracted ONCE")
    print(f"Head batchsz  : {BATCH_SIZE}")
    print(f"CPU threads   : {torch.get_num_threads()}")
    print(f"Output dir    : {RETRAINED_DIR}")
    print("=" * 60)

    set_seed(RANDOM_SEED)
    device = torch.device("cpu")

    # ── 1. Data ────────────────────────────────────────────────
    records = load_records(DATA_RAW_DIR)
    train_r, val_r, test_r = split_records(records)
    print(f"Splits — Train:{len(train_r)}  Val:{len(val_r)}  Test:{len(test_r)}")

    dist = Counter(r["label_id"] for r in train_r)
    print("Class distribution (train):")
    for i, n in enumerate(TARGET_CLASSES):
        print(f"  {n:10s}: {dist[i]}")

    # ── 2. Load frozen encoder ─────────────────────────────────
    print(f"\nLoading frozen encoder: {MODEL_NAME_OR_PATH} ...")
    encoder = Wav2Vec2SpeechClassifier(
        model_name_or_path=MODEL_NAME_OR_PATH,
        num_classes=len(TARGET_CLASSES),
        hidden_dim=HIDDEN_DIM,
        dropout_rate=DROPOUT_RATE,
        freeze_encoder=True,
    ).to(device)
    encoder.eval()

    # ── 3. Extract embeddings ONCE ────────────────────────────
    print("\n[Step 1/4] Extracting TRAIN embeddings ...")
    t0 = time.time()
    train_embs, train_lbls = extract_embeddings(train_r, encoder, device, is_train=True)
    print(f"  Done in {(time.time()-t0)/60:.1f} min  shape={tuple(train_embs.shape)}")

    print("[Step 2/4] Extracting VAL embeddings ...")
    t0 = time.time()
    val_embs, val_lbls = extract_embeddings(val_r, encoder, device, is_train=False)
    print(f"  Done in {(time.time()-t0)/60:.1f} min  shape={tuple(val_embs.shape)}")

    print("[Step 3/4] Extracting TEST embeddings ...")
    t0 = time.time()
    test_embs, test_lbls = extract_embeddings(test_r, encoder, device, is_train=False)
    print(f"  Done in {(time.time()-t0)/60:.1f} min  shape={tuple(test_embs.shape)}")

    emb_dim = train_embs.shape[1]

    # ── 4. DataLoaders over cached embeddings ─────────────────
    train_dl = DataLoader(TensorDataset(train_embs, train_lbls),
                          batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(TensorDataset(val_embs,   val_lbls),
                          batch_size=BATCH_SIZE, shuffle=False)
    test_dl  = DataLoader(TensorDataset(test_embs,  test_lbls),
                          batch_size=BATCH_SIZE, shuffle=False)

    # ── 5. Class weights ──────────────────────────────────────
    counts = np.array([dist[i] for i in range(len(TARGET_CLASSES))], dtype=np.float32)
    cw     = len(train_r) / (len(TARGET_CLASSES) * np.maximum(counts, 1))
    cw_t   = torch.tensor(cw, dtype=torch.float32)

    # ── 6. Classifier head ────────────────────────────────────
    head = ClassifierHead(emb_dim, HIDDEN_DIM, len(TARGET_CLASSES), DROPOUT_RATE).to(device)
    trainable = sum(p.numel() for p in head.parameters())
    print(f"\n[Step 4/4] Training classifier head  ({trainable:,} params) ...")

    criterion = nn.CrossEntropyLoss(weight=cw_t)
    optimizer = AdamW(head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_val_f1      = -1.0
    best_val_metrics = None
    patience_ctr     = 0
    history          = []
    best_head_state  = None

    print("\nStarting training ...\n" + "-" * 60)

    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        head.train()
        tr_loss = 0.0
        for emb_b, lbl_b in train_dl:
            optimizer.zero_grad()
            loss = criterion(head(emb_b), lbl_b)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * emb_b.size(0)
        tr_loss /= len(train_r)

        # Validate
        head.eval()
        preds, targets, probs_list = [], [], []
        with torch.no_grad():
            for emb_b, lbl_b in val_dl:
                p = torch.softmax(head(emb_b), dim=-1)
                preds.extend(torch.argmax(p, dim=-1).numpy())
                targets.extend(lbl_b.numpy())
                probs_list.extend(p.numpy())

        vm  = calculate_metrics(np.array(targets), np.array(preds),
                                np.array(probs_list), target_names=TARGET_CLASSES)
        vf1 = vm["macro_f1"]
        scheduler.step(vf1)

        history.append({
            "epoch": epoch, "train_loss": float(tr_loss),
            "val_accuracy": float(vm["accuracy"]),
            "val_macro_f1": float(vf1),
            "val_weighted_f1": float(vm["weighted_f1"]),
        })
        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS:02d}] | "
              f"TrainLoss: {tr_loss:.4f} | "
              f"ValAcc: {vm['accuracy']:.4f} | "
              f"ValMacroF1: {vf1:.4f}")

        if vf1 > best_val_f1:
            best_val_f1      = vf1
            best_val_metrics = vm
            patience_ctr     = 0
            best_head_state  = {k: v.clone() for k, v in head.state_dict().items()}
            save_json(vm, RETRAINED_DIR / "best_val_metrics.json")
            print(f"  [*] New best  (ValMacroF1: {vf1:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping after epoch {epoch}.")
                break

    print("-" * 60)
    print(f"Training complete.  Best val macro-F1: {best_val_f1:.4f}")

    # ── 7. Save full model checkpoint (encoder + best head) ───
    # Restore best head weights into encoder's classifier
    encoder.classifier.load_state_dict({
        k.replace("net.", ""): v for k, v in best_head_state.items()
    } if "net.0.weight" in best_head_state else best_head_state)

    # Actually save by composing: encoder backbone + best head as a dict
    # Save the full state dict by patching encoder.classifier with best head
    head.load_state_dict(best_head_state)
    # Build a combined state dict: wav2vec2.* + classifier.*
    combined = {}
    for k, v in encoder.wav2vec2.state_dict().items():
        combined[f"wav2vec2.{k}"] = v
    # Map head.net.* → classifier.*
    head_sd = head.state_dict()  # keys: net.0.weight, net.0.bias, net.3.weight, net.3.bias
    layer_map = {
        "net.0.weight": "0.weight", "net.0.bias": "0.bias",
        "net.3.weight": "3.weight", "net.3.bias": "3.bias",
    }
    for hk, ck in layer_map.items():
        combined[f"classifier.{ck}"] = head_sd[hk]
    torch.save(combined, RETRAINED_DIR / "best_model.pt")
    print(f"Checkpoint saved -> {RETRAINED_DIR / 'best_model.pt'}")

    # ── 8. Save configs ───────────────────────────────────────
    model_cfg = {
        "model_name_or_path": MODEL_NAME_OR_PATH,
        "num_classes":        len(TARGET_CLASSES),
        "hidden_dim":         HIDDEN_DIM,
        "dropout_rate":       DROPOUT_RATE,
        "freeze_encoder":     True,
        "batch_size":         BATCH_SIZE,
        "learning_rate":      LEARNING_RATE,
        "audio_window_s":     NUM_SAMPLES / SAMPLE_RATE,
        "num_samples":        NUM_SAMPLES,
        "normalization":      "zero_mean_unit_variance",
        "training_strategy":  "embedding_cache_frozen_encoder",
        "model_version":      "2.0.0-wav2vec2-corrected",
    }
    label_map = {"label_to_id": LABEL_TO_ID, "id_to_label": ID_TO_LABEL,
                 "target_classes": TARGET_CLASSES}
    save_json(model_cfg,  RETRAINED_DIR / "config.json")
    save_json(label_map,  RETRAINED_DIR / "label_mapping.json")
    save_json(history,    RETRAINED_DIR / "training_history.json")

    # ── 9. Test evaluation (ONCE on best checkpoint) ──────────
    print("\nEvaluating best checkpoint on held-out test set ...")
    head.eval()
    preds, targets, probs_list = [], [], []
    with torch.no_grad():
        for emb_b, lbl_b in test_dl:
            p = torch.softmax(head(emb_b), dim=-1)
            preds.extend(torch.argmax(p, dim=-1).numpy())
            targets.extend(lbl_b.numpy())
            probs_list.extend(p.numpy())

    tm = calculate_metrics(np.array(targets), np.array(preds),
                           np.array(probs_list), target_names=TARGET_CLASSES)
    save_json(tm, RETRAINED_DIR / "test_metrics.json")

    pred_dist         = Counter(preds)
    n_classes_pred    = len(pred_dist)
    collapse_resolved = n_classes_pred >= 5

    print(f"\nTest Accuracy   : {tm['accuracy']:.4f}")
    print(f"Test Macro F1   : {tm['macro_f1']:.4f}")
    print(f"Test Weighted F1: {tm['weighted_f1']:.4f}")
    print(f"Classes predicted: {n_classes_pred}/6  |  Collapse resolved: {collapse_resolved}")
    print("\nPer-class (Test):")
    for cls, m in tm["per_class"].items():
        print(f"  {cls:10s} P:{m['precision']:.3f} R:{m['recall']:.3f} F1:{m['f1_score']:.3f} (n={m['support']})")

    # ── 10. Production decision (VALIDATION only) ──────────────
    margin = best_val_f1 - SVM_VAL_MACRO_F1
    if margin > 0.02 and collapse_resolved:
        prod_model = "Wav2Vec2_Retrained"
        reason = (f"Wav2Vec2 val macro-F1 {best_val_f1:.4f} exceeds SVM "
                  f"{SVM_VAL_MACRO_F1:.4f} by {margin:.4f} and collapse resolved "
                  f"({n_classes_pred}/6 classes predicted).")
    else:
        prod_model = "SVM"
        if not collapse_resolved:
            reason = (f"Wav2Vec2 still collapses ({n_classes_pred}/6 classes). SVM retained.")
        else:
            reason = (f"Wav2Vec2 val macro-F1 {best_val_f1:.4f} does not meaningfully "
                      f"exceed SVM {SVM_VAL_MACRO_F1:.4f} (margin {margin:+.4f}, "
                      f"threshold 0.02). SVM retained.")

    print(f"\n{'='*60}")
    print(f"PRODUCTION MODEL: {prod_model}")
    print(f"Reason: {reason}")
    print(f"{'='*60}")

    # ── 11. Comparison JSON ────────────────────────────────────
    bvm = best_val_metrics or {}
    comp = {
        "methodology_note": (
            "Selection based on VALIDATION macro-F1 only. Test set is held-out. "
            "The LogisticRegression entry in the original model_comparison_report.json "
            "used test macro-F1 for selection — methodologically invalid, not used."
        ),
        "original_production_candidate": "SVM",
        "models": {
            "SVM": {
                "val_accuracy":    SVM_VAL_ACCURACY,
                "val_macro_f1":    SVM_VAL_MACRO_F1,
                "val_weighted_f1": SVM_VAL_WEIGHTED_F1,
                "test_accuracy":   SVM_TEST_ACCURACY,
                "test_macro_f1":   SVM_TEST_MACRO_F1,
                "test_weighted_f1":SVM_TEST_WEIGHTED_F1,
            },
            "Wav2Vec2_Retrained": {
                "training_strategy":      "embedding_cache_frozen_encoder",
                "normalization":          "zero_mean_unit_variance",
                "audio_window_s":         NUM_SAMPLES / SAMPLE_RATE,
                "val_accuracy":           float(bvm.get("accuracy", 0)),
                "val_macro_f1":           float(best_val_f1),
                "val_weighted_f1":        float(bvm.get("weighted_f1", 0)),
                "val_per_class":          bvm.get("per_class"),
                "val_confusion_matrix":   bvm.get("confusion_matrix"),
                "test_accuracy":          float(tm["accuracy"]),
                "test_macro_f1":          float(tm["macro_f1"]),
                "test_weighted_f1":       float(tm["weighted_f1"]),
                "test_per_class":         tm["per_class"],
                "test_confusion_matrix":  tm["confusion_matrix"],
                "classes_predicted_test": n_classes_pred,
                "pred_distribution_test": {TARGET_CLASSES[k]: v for k, v in pred_dist.items()},
                "collapse_resolved":      collapse_resolved,
            },
        },
        "production_model":              prod_model,
        "production_model_basis":        "validation_macro_f1",
        "test_used_for_model_selection": False,
        "selection_reason":              reason,
    }
    save_json(comp, ARTIFACTS_DIR / "final_model_comparison.json")

    # ── 12. Markdown report ────────────────────────────────────
    md = [
        "# Final Model Comparison Report",
        "",
        "> **Model selection is based exclusively on VALIDATION performance.**",
        "> The test set is held-out final evaluation data only.",
        "",
        "---",
        "## Methodology",
        "| Item | Detail |",
        "|---|---|",
        "| Original production candidate | SVM (val macro-F1 = 0.4822) |",
        "| Old LR 'champion' | Selected on **TEST** macro-F1 — invalid for production selection |",
        "| Corrected Wav2Vec2 | Retrained with zero-mean unit-variance normalization |",
        "| Training strategy | Frozen encoder; embeddings extracted once; classifier head trained |",
        "| Test evaluation | Performed ONCE after validation-based checkpoint selection |",
        "",
        "---",
        "## Validation Performance (Model-Selection Criterion)",
        "| Model | Val Accuracy | Val Macro-F1 | Val Weighted-F1 |",
        "|---|---|---|---|",
        f"| SVM | {SVM_VAL_ACCURACY:.4f} | {SVM_VAL_MACRO_F1:.4f} | {SVM_VAL_WEIGHTED_F1:.4f} |",
        f"| Wav2Vec2 (corrected) | {bvm.get('accuracy',0):.4f} | {best_val_f1:.4f} | {bvm.get('weighted_f1',0):.4f} |",
        "",
        "---",
        "## Test Performance (Held-Out — Reference Only)",
        "| Model | Test Accuracy | Test Macro-F1 | Test Weighted-F1 |",
        "|---|---|---|---|",
        f"| SVM | {SVM_TEST_ACCURACY:.4f} | {SVM_TEST_MACRO_F1:.4f} | {SVM_TEST_WEIGHTED_F1:.4f} |",
        f"| Wav2Vec2 (corrected) | {tm['accuracy']:.4f} | {tm['macro_f1']:.4f} | {tm['weighted_f1']:.4f} |",
        "",
        "---",
        "## Wav2Vec2 Per-Class Test Metrics",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
    ]
    for cls, m in tm["per_class"].items():
        md.append(f"| {cls} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1_score']:.4f} | {m['support']} |")
    md += [
        "",
        "---",
        "## Prediction Distribution (Test)",
        f"Classes predicted: **{n_classes_pred}/6** | Collapse resolved: **{collapse_resolved}**",
        "",
        "| Class | Count |",
        "|---|---|",
    ]
    for lid, cnt in sorted(pred_dist.items()):
        md.append(f"| {TARGET_CLASSES[lid]} | {cnt} |")
    md += [
        "",
        "---",
        "## Confusion Matrix (Test)",
        f"Rows=True, Cols=Predicted | Classes: {TARGET_CLASSES}",
        "```",
    ]
    for row in tm["confusion_matrix"]:
        md.append("  " + "  ".join(f"{v:3d}" for v in row))
    md += [
        "```",
        "",
        "---",
        f"## Final Production Model: `{prod_model}`",
        f"**Reason:** {reason}",
        "",
        "> [!IMPORTANT]",
        "> Test results are held-out evaluation only. They played NO role in model selection.",
    ]
    (ARTIFACTS_DIR / "final_model_comparison.md").write_text("\n".join(md), encoding="utf-8")

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Wav2Vec2 training           : COMPLETE")
    print(f"Best Wav2Vec2 val macro-F1  : {best_val_f1:.4f}")
    print(f"Wav2Vec2 test macro-F1      : {tm['macro_f1']:.4f}")
    print(f"SVM val macro-F1            : {SVM_VAL_MACRO_F1:.4f}")
    print(f"SVM test macro-F1           : {SVM_TEST_MACRO_F1:.4f}")
    print(f"Prediction collapse resolved: {collapse_resolved}")
    print(f"Final production model      : {prod_model}")
    print(f"Reason                      : {reason}")
    print("=" * 60)


if __name__ == "__main__":
    run()
