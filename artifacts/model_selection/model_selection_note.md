# Model Selection Methodology Note

## Production Model: SVM

**Validation macro-F1: 0.4822**

## Why LogisticRegression in the old comparison report is not the production model

The file `artifacts/wav2vec2/model_comparison_report.json` identifies LogisticRegression as the
`"champion_model"` with a macro-F1 of 0.3853. That score is the **test set** macro-F1, not
validation.

Using test performance for model selection is methodologically incorrect because:

1. **The test set must remain a held-out, unbiased final evaluation set.** Selecting a model based
   on test performance turns it into a second validation set, destroying its purpose as an
   independent estimate of generalization.

2. **Validation performance is the correct basis for model selection.** On the validation set
   (speakers 17–20, fully disjoint from training speakers 1–16), the ranking is:

   | Model              | Val macro-F1 |
   |--------------------|-------------|
   | **SVM**            | **0.4822**  |
   | RandomForest       | 0.4160      |
   | LogisticRegression | 0.3566      |

   SVM is the clear winner by all three validation metrics (macro-F1, weighted-F1, accuracy).

3. **The old comparison report's champion field was computed over test data**, which is why it
   disagrees with the validation-based selection in `model_config.json`. The `model_config.json`
   entry (`"selected_model": "SVM"`) is correct.

## Resolution

- `artifacts/baseline/model_config.json` → SVM ✓ (unchanged, already correct)
- `artifacts/baseline/best_model.joblib` → SVM (SVC) ✓ (verified by deserialization)
- `predict_baseline.py` → loads from `best_model.joblib` ✓ (consistent)
- `artifacts/model_selection/final_model_selection.json` → created as canonical selection record
- `artifacts/wav2vec2/model_comparison_report.json` → **not modified**; retained as historical record

No model artifacts were changed. No retraining was performed.
