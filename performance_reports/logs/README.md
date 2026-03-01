# Logs

Contains timestamped execution logs from training, evaluation, and validation scripts.

## Files

### Training Logs
- `train_YYYYMMDD_HHMMSS.log` - Training progress, epoch metrics, model checkpoints, early stopping events

### Evaluation Logs
- `evaluate_YYYYMMDD_HHMMSS.log` - Evaluation process, accuracy metrics, inference times

### Validation Logs
- `validate_YYYYMMDD_HHMMSS.log` - Validation accuracy, classification metrics

## Log Format
```
2026-03-01 14:30:22 - INFO - Starting Training
2026-03-01 14:30:25 - INFO - Epoch 1/50 - Train Loss: 0.8750, Train Acc: 0.6540...
2026-03-01 14:32:10 - INFO - Best model saved at epoch 10 with Val Acc: 0.8923
```

## Purpose
- Track training/evaluation progress
- Debug issues and errors
- Audit model performance over time
- Review hyperparameters and configurations

## Output
Logs are written to both the log file and console (stdout)
