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
from model.inference.predict import CryingSensePredictor

predictor = CryingSensePredictor('model/saved_models/cryingsense_cnn_best.pth')
result = predictor.predict_single('audio.wav')
print(f"Predicted: {result['prediction']} ({result['confidence']:.2%})")
```

### Inference with Database Integration

Store classifications, audio sessions, and audio files automatically:

```python
from model.inference.predict import CryingSensePredictor

# Initialize with database enabled
predictor = CryingSensePredictor(
    model_path='saved_models/cryingsense_cnn_best.pth',
    save_to_db=True,
    device_id='ESP32-001',
    device_source='esp32'
)

# Start monitoring session
session_id = predictor.start_new_session()

# Perform inference (automatically saves to database)
result = predictor.predict_single('baby_cry.wav')
print(f"Classification ID: {result.get('classification_id')}")
print(f"Audio File ID: {result.get('audio_file_id')}")

# End session
predictor.end_current_session()
```

**See:** [Database Integration Guide](inference/DATABASE_INTEGRATION.md) for detailed documentation

### Command-Line Inference with Database

```bash
# Single file with database storage
python -m model.inference.predict \
  --audio test.wav \
  --model saved_models/cryingsense_cnn.pth \
  --save-to-db \
  --device-id ESP32-001 \
  --device-source esp32

# With session tracking
python -m model.inference.predict \
  --audio test.wav \
  --model saved_models/cryingsense_cnn.pth \
  --save-to-db \
  --device-id ESP32-001 \
  --session-id monitoring-001
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
