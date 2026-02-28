# CryingSense Model

CNN-based infant cry classification model for detecting and categorizing baby cries.

## Directory Structure

```
model/
├── models/              # Model architecture & export
│   ├── cnn_model.py     # CryingSenseCNN architecture
│   └── export_model.py  # Export to TorchScript, ONNX, quantized
├── inference/           # Prediction pipeline
│   ├── audio_preprocessor.py  # Audio loading & preprocessing
│   ├── feature_extractor.py   # MFCC, Mel, Chroma extraction
│   ├── model_loader.py        # Load trained models
│   └── predict.py             # Run inference
├── training/            # Training pipeline
│   ├── train.py         # Main training script
│   ├── evaluate.py      # Model evaluation
│   └── validate.py      # Validation utilities
├── saved_models/        # Trained model checkpoints
│   ├── cryingsense_cnn_best.pth  # Best trained model
│   └── exported/        # Exported deployment models
└── requirements.txt     # Python dependencies
```

## Model Architecture

**CryingSenseCNN** - Lightweight CNN with depthwise separable convolutions:

- **Input:** 4-channel feature tensor (128 x 216)
  - Channel 0: MFCC (40 coefficients)
  - Channel 1: Mel spectrogram (128 bins)
  - Channel 2: Chroma features (12 bins)
  - Channel 3: Delta MFCC
- **Output:** 6 classes (belly_pain, burp, discomfort, hunger, noise, tired)
- **Parameters:** ~12,410
- **Size:** ~0.05 MB

## Quick Start

### Training

```bash
cd model
python -m training.train --data-dir ../dataset --epochs 50
```

### Evaluation

```bash
python -m training.evaluate --model saved_models/cryingsense_cnn_best.pth
```

### Inference

```python
from model.inference.predict import CryPredictor

predictor = CryPredictor('model/saved_models/cryingsense_cnn_best.pth')
result = predictor.predict('audio.wav')
print(f"Predicted: {result['class']} ({result['confidence']:.2%})")
```

### Export for Deployment

```bash
python -m model.models.export_model
# Creates: cryingsense_model_beta_v1.torchscript.pt, .pth, _quantized.pth
```

## Audio Requirements

| Parameter | Value |
|-----------|-------|
| Sample Rate | 16,000 Hz |
| Duration | 5 seconds |
| Channels | Mono |
| Format | WAV (recommended) |

## Classes

| Index | Class | Description |
|-------|-------|-------------|
| 0 | belly_pain | Colic or stomach discomfort |
| 1 | burp | Needs burping |
| 2 | discomfort | General discomfort |
| 3 | hunger | Hungry cry |
| 4 | noise | Non-cry sounds |
| 5 | tired | Sleepy/tired cry |

## Training Results

Current best model (`cryingsense_cnn_best.pth`):
- **Epoch:** 29
- **Validation Accuracy:** 59.24%

## Dependencies

```bash
pip install -r requirements.txt
```

Key packages:
- PyTorch >= 2.0
- librosa
- numpy
- soundfile
