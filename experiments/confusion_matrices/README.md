# Confusion Matrices

This directory stores confusion matrices from model evaluations.

## Purpose
Visual representation of model performance showing:
- True vs predicted classifications
- Class-specific accuracy
- Common misclassification patterns

## File Naming Convention
```
confusion_matrix_{model}_{dataset}_{date}.png
```

Example: `confusion_matrix_cnn_baseline_test_20240204.png`

## Generation
Confusion matrices are generated during:
- Model evaluation (`model/training/evaluate.py`)
- Validation checkpoints
- Final test set evaluation

## Interpretation
- Diagonal elements: Correct predictions
- Off-diagonal elements: Misclassifications
- Row: True labels
- Column: Predicted labels

Values are typically normalized to show percentages.
