# Experiment Logs

This directory contains training logs and experiment tracking data.

## Contents
- Training loss and accuracy logs
- Validation metrics per epoch
- Hyperparameter configurations
- Model checkpoints

## File Naming Convention
```
experiment_{date}_{time}_{description}.log
```

Example: `experiment_20240204_143022_baseline_cnn.log`

## Log Format
Logs are stored in JSON Lines format for easy parsing:
```json
{"epoch": 1, "train_loss": 0.875, "train_acc": 0.654, "val_loss": 0.923, "val_acc": 0.612, "lr": 0.001}
{"epoch": 2, "train_loss": 0.743, "train_acc": 0.723, "val_loss": 0.812, "val_acc": 0.687, "lr": 0.001}
```

## Tools
Use provided scripts to analyze logs:
- `../scripts/plot_training_curves.py` (to be implemented)
- `../scripts/compare_experiments.py` (to be implemented)
