# Final Model Comparison Report

> **Model selection is based exclusively on VALIDATION performance.**
> The test set is held-out final evaluation data only.

---
## Methodology
| Item | Detail |
|---|---|
| Original production candidate | SVM (val macro-F1 = 0.4822) |
| Old LR 'champion' | Selected on **TEST** macro-F1 — invalid for production selection |
| Corrected Wav2Vec2 | Retrained with zero-mean unit-variance normalization |
| Training strategy | Frozen encoder; embeddings extracted once; classifier head trained |
| Test evaluation | Performed ONCE after validation-based checkpoint selection |

---
## Validation Performance (Model-Selection Criterion)
| Model | Val Accuracy | Val Macro-F1 | Val Weighted-F1 |
|---|---|---|---|
| SVM | 0.4861 | 0.4822 | 0.4822 |
| Wav2Vec2 (corrected) | 0.2083 | 0.0984 | 0.0984 |

---
## Test Performance (Held-Out — Reference Only)
| Model | Test Accuracy | Test Macro-F1 | Test Weighted-F1 |
|---|---|---|---|
| SVM | 0.3472 | 0.3348 | 0.3348 |
| Wav2Vec2 (corrected) | 0.1528 | 0.0625 | 0.0625 |

---
## Wav2Vec2 Per-Class Test Metrics
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| neutral | 0.0000 | 0.0000 | 0.0000 | 12 |
| happy | 0.2500 | 0.0833 | 0.1250 | 12 |
| sad | 0.0000 | 0.0000 | 0.0000 | 12 |
| angry | 0.1471 | 0.8333 | 0.2500 | 12 |
| anxious | 0.0000 | 0.0000 | 0.0000 | 12 |
| distress | 0.0000 | 0.0000 | 0.0000 | 12 |

---
## Prediction Distribution (Test)
Classes predicted: **2/6** | Collapse resolved: **False**

| Class | Count |
|---|---|
| happy | 4 |
| angry | 68 |

---
## Confusion Matrix (Test)
Rows=True, Cols=Predicted | Classes: ['neutral', 'happy', 'sad', 'angry', 'anxious', 'distress']
```
    0    0    0   12    0    0
    0    1    0   11    0    0
    0    0    0   12    0    0
    0    2    0   10    0    0
    0    0    0   12    0    0
    0    1    0   11    0    0
```

---
## Final Production Model: `SVM`
**Reason:** Wav2Vec2 still collapses (2/6 classes). SVM retained.

> [!IMPORTANT]
> Test results are held-out evaluation only. They played NO role in model selection.