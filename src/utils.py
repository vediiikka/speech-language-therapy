import os
import random
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    f1_score
)

def set_seed(seed: int = 42):
    """Set random seeds for complete reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def calculate_metrics(y_true, y_pred, y_prob=None, target_names=None):
    """
    Calculate comprehensive clinical evaluation metrics:
    - Accuracy
    - Macro Precision, Recall, F1
    - Weighted F1
    - Per-class Precision, Recall, F1
    - Confusion Matrix
    """
    if target_names is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
        target_names = [str(l) for l in labels]
    else:
        labels = list(range(len(target_names)))

    acc = float(accuracy_score(y_true, y_pred))
    
    # Macro metrics
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # Weighted metrics
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    per_class_p, per_class_r, per_class_f1, per_class_support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=labels, zero_division=0
    )

    per_class_dict = {}
    for idx, name in enumerate(target_names):
        per_class_dict[name] = {
            "precision": float(per_class_p[idx]),
            "recall": float(per_class_r[idx]),
            "f1_score": float(per_class_f1[idx]),
            "support": int(per_class_support[idx])
        }

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return {
        "accuracy": acc,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class_dict,
        "confusion_matrix": cm,
        "class_labels": target_names
    }


def save_json(data: dict, file_path: Path):
    """Save dictionary as formatted JSON file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


def load_json(file_path: Path) -> dict:
    """Load JSON file into dictionary."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
