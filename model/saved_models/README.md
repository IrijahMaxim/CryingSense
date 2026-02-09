# Trained Models

This directory will contain trained model checkpoints.

## Files Generated During Training

- `cryingsense_cnn_best.pth` - Best model checkpoint from training
- `training_history.json` - Training metrics history
- `training_curves.png` - Training visualization plots

## Exported Models

When using `export_model.py`, models are exported to `exported/` subdirectory:

- `cryingsense_cnn.torchscript.pt` - TorchScript format (optimized)
- `cryingsense_cnn.onnx` - ONNX format (cross-platform)
- `cryingsense_cnn_quantized.pth` - Quantized model (reduced size)

## Important Notes

- Model files (*.pth, *.pt) are **not committed** to git (see .gitignore)
- To use a model, you must:
  1. Train it yourself using `model/training/train.py`, OR
  2. Obtain a pre-trained checkpoint from project releases
- Model checkpoints include training metadata (epoch, accuracy, etc.)

## Training a Model

```bash
cd ../training
python train.py
```

See QUICK_START.md for detailed instructions.
