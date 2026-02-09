# CryingSense - Quick Start Guide

This guide will help you get the CryingSense baby cry classification system up and running.

## Prerequisites

- Python 3.8 or higher
- PyTorch 2.6.0+
- Audio input device (microphone)

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/IrijahMaxim/CryingSense.git
   cd CryingSense
   ```

2. **Install dependencies**
   ```bash
   pip install -r model/requirements.txt
   ```

3. **Install audio dependencies** (for microphone recording)
   
   **Linux:**
   ```bash
   sudo apt-get install portaudio19-dev
   pip install pyaudio
   ```
   
   **macOS:**
   ```bash
   brew install portaudio
   pip install pyaudio
   ```
   
   **Windows:**
   ```bash
   pip install pyaudio
   ```

## Step-by-Step Workflow

### Step 1: Prepare Your Dataset

Organize your cry audio files in the following structure:

```
dataset/raw/
├── belly_pain/
│   ├── cry_001.wav
│   ├── cry_002.wav
│   └── ...
├── burp/
│   └── ...
├── discomfort/
│   └── ...
├── hunger/
│   └── ...
└── tired/
    └── ...
```

### Step 2: Preprocess Audio

```bash
cd scripts
python preprocess_audio.py
```

This will:
- Remove noise
- Trim silence
- Normalize amplitude
- Create 5-second segments

Output: `dataset/processed/cleaned/`

### Step 3: Extract Features

```bash
python feature_extraction.py
```

This extracts:
- MFCC (40 coefficients)
- Mel Spectrogram (128 bands)
- Chroma (12 bins)

Output: `dataset/processed/features/`

### Step 4: Split Dataset

```bash
python dataset_split.py
```

This creates an 80/10/10 train/validation/test split.

Output: `dataset/dataset_split.json`

### Step 5: Train the Model

```bash
cd ../model/training
python train.py
```

Training features:
- Early stopping (patience: 10 epochs)
- Learning rate scheduling
- Data augmentation
- Automatic checkpointing

Training time: Depends on dataset size and hardware
- GPU: ~10-30 minutes for small datasets
- CPU: ~1-3 hours for small datasets

Output: `model/saved_models/cryingsense_cnn_best.pth`

### Step 6: Evaluate the Model

```bash
python evaluate.py
```

This generates:
- Overall accuracy, precision, recall, F1-score
- Per-class metrics
- Confusion matrix visualization
- Inference time measurements

Output: `experiments/performance_reports/` and `experiments/confusion_matrices/`

### Step 7: Export for Deployment (Optional)

```bash
cd ../models
python export_model.py --model ../saved_models/cryingsense_cnn_best.pth
```

This creates:
- TorchScript model (`.torchscript.pt`)
- ONNX model (`.onnx`)
- Quantized model (reduced size)

Output: `model/saved_models/exported/`

### Step 8: Test with Live Audio

**Record audio:**
```bash
cd ../../audio_test
python record_audio.py --duration 5 --output recordings
```

**Test with model:**
```bash
python test_live.py \
  --model ../model/saved_models/cryingsense_cnn_best.pth \
  --audio recordings/ \
  --threshold 0.6
```

This will:
- Load the trained model
- Process each audio file
- Extract features
- Run inference
- Display predictions with confidence scores

## Example Output

```
==============================================================================
PREDICTION RESULTS
==============================================================================
Audio File: recording_20240209_120530.wav
Timestamp: 2024-02-09T12:05:35.123456

Prediction: HUNGER
Confidence: 87.50%
Status: ✓ CONFIDENT

Class Probabilities:
  hunger          [████████████████████████████████████░░░░] 87.50%
  discomfort      [███████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 6.30%
  tiredness       [██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 3.20%
  belly_pain      [█░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 2.00%
  burp            [█░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 1.00%

Timing:
  Feature Extraction: 145.23 ms
  Model Inference:    23.45 ms
  Total Time:         168.68 ms
==============================================================================
```

## Continuous Recording and Testing

For real-time testing, you can record continuously:

**Terminal 1 - Continuous Recording:**
```bash
cd audio_test
python record_audio.py --continuous --duration 5 --output recordings
```

**Terminal 2 - Test New Recordings:**
```bash
cd audio_test
# Wait for recordings to be created, then test
python test_live.py --model ../model/saved_models/cryingsense_cnn_best.pth --audio recordings/recording_latest.wav
```

## Model Performance

Expected performance on well-prepared dataset:
- **Accuracy**: 85-90%
- **Inference Time**: 20-50ms on CPU, 5-10ms on GPU
- **Model Size**: ~8 MB (fp32), ~2 MB (quantized)
- **Parameters**: ~1-2 million

## Troubleshooting

### Issue: "Model not found"
**Solution**: Train the model first using Step 5 above.

### Issue: "No audio files found"
**Solution**: Make sure audio files are in `.wav`, `.mp3`, or `.flac` format.

### Issue: "Low confidence predictions"
**Solutions**:
- Ensure audio quality is good (clear, not too noisy)
- Check that audio matches training data characteristics
- Adjust confidence threshold: `--threshold 0.5` (lower) or `--threshold 0.7` (higher)

### Issue: "Feature extraction errors"
**Solution**: Ensure audio files are:
- Valid format (WAV, MP3, FLAC)
- Not corrupted
- Have audio data (not silent)

### Issue: "PyAudio installation fails"
**Solution**: Install system audio libraries first (see Installation section above)

## Advanced Usage

### Adjust Training Parameters

Edit `model/training/train.py` to customize:
- Batch size
- Learning rate
- Number of epochs
- Early stopping patience
- Dropout rate

### Modify Model Architecture

Edit `model/models/cnn_model.py` to:
- Add/remove convolutional layers
- Change channel dimensions
- Adjust pooling strategy

### Custom Feature Extraction

Edit `scripts/feature_extraction.py` to:
- Change MFCC coefficients
- Adjust Mel bands
- Modify time resolution

## Production Deployment

For deploying on Raspberry Pi or edge devices:

1. **Use quantized model** for smaller size and faster inference
2. **Export to TorchScript** for optimized deployment
3. **Test inference time** on target hardware
4. **Set appropriate confidence threshold** for your use case

See `iot/raspberry_pi/README.md` for detailed deployment instructions.

## Project Structure

```
CryingSense/
├── model/                    # ML models and training
│   ├── models/              # Model architectures
│   ├── training/            # Training scripts
│   ├── saved_models/        # Trained model checkpoints
│   └── MODEL_REQUIREMENTS.md
├── scripts/                  # Data processing
│   ├── preprocess_audio.py
│   ├── feature_extraction.py
│   └── dataset_split.py
├── audio_test/              # Audio recording and testing
│   ├── record_audio.py
│   ├── test_live.py
│   └── README.md
├── dataset/                 # Audio dataset
│   ├── raw/                # Original recordings
│   └── processed/          # Processed features
├── experiments/            # Training results
│   ├── logs/
│   ├── confusion_matrices/
│   └── performance_reports/
└── iot/                    # IoT deployment code
    └── raspberry_pi/
```

## Next Steps

1. **Collect More Data**: Improve model accuracy with more training samples
2. **Fine-tune Hyperparameters**: Experiment with learning rates, batch sizes
3. **Deploy to Edge Device**: Set up Raspberry Pi inference
4. **Build Mobile App**: Create user-friendly interface
5. **Monitor Performance**: Track real-world accuracy

## Getting Help

- **Documentation**: See README files in each directory
- **Requirements**: See `model/MODEL_REQUIREMENTS.md` for detailed compliance
- **Issues**: Open an issue on GitHub
- **Code**: All code is well-commented with docstrings

## License

This project is licensed under the MIT License. See LICENSE file for details.

---

**Quick Start Version**: 1.0  
**Last Updated**: 2024-02-09  
**Status**: Ready to use ✅
