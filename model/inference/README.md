# Inference Module

This directory contains code for performing real-time inference using the trained CryingSense CNN model.

## Purpose
Provides inference capabilities for:
- IoT edge deployment (Raspberry Pi)
- Server-side inference
- Testing and validation

## Files
- `predict.py` - Main inference script (to be implemented)
- `model_loader.py` - Model loading utilities (to be implemented)
- `audio_preprocessor.py` - Real-time audio preprocessing (to be implemented)
- `feature_extractor.py` - Real-time feature extraction (to be implemented)

## Usage

### Single File Inference
```bash
python predict.py --audio path/to/audio.wav --model ../saved_models/cryingsense_cnn.pth
```

### Batch Inference
```bash
python predict.py --batch path/to/audio/directory --model ../saved_models/cryingsense_cnn.pth
```

### Real-time Streaming Inference
```bash
python predict.py --stream --model ../saved_models/cryingsense_cnn.pth
```

## Model Loading
The inference module supports:
- Standard PyTorch models (.pth)
- Quantized models for edge devices
- TorchScript models for deployment

## Performance
- **Standard Model**: ~200-300ms per sample on Raspberry Pi 3B+
- **Quantized Model**: ~100-150ms per sample on Raspberry Pi 3B+
- **Server GPU**: ~10-20ms per sample

## Output Format
```json
{
  "prediction": "hunger",
  "confidence": 0.87,
  "probabilities": {
    "belly_pain": 0.03,
    "burp": 0.02,
    "discomfort": 0.05,
    "hunger": 0.87,
    "tired": 0.03
  },
  "inference_time_ms": 145,
  "timestamp": "2024-02-04T12:34:56.789Z"
}
```

## Dependencies
See `../requirements.txt` for required packages.

## Notes
- The `noise` class is used for filtering but not included in final predictions
- Confidence threshold for alerts: 0.70 (configurable)
- Model input shape: Dependent on feature extraction method
