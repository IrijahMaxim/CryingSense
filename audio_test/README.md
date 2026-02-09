# Audio Test Module

This module provides tools for testing the CryingSense CNN model with live audio recordings from your system microphone.

## Features

- **Audio Recording**: Record audio from system default microphone
- **Live Testing**: Test trained CNN model on recorded audio
- **Real-time Inference**: Get instant predictions with confidence scores
- **Continuous Recording**: Record multiple audio segments automatically

## Files

- `record_audio.py` - Audio recording utility
- `test_live.py` - Live testing and inference script
- `README.md` - This file

## Quick Start

### 1. List Available Microphones

```bash
python record_audio.py --list-devices
```

### 2. Record Audio

Record a 5-second audio clip:
```bash
python record_audio.py --duration 5 --output recordings
```

Record continuously (press Ctrl+C to stop):
```bash
python record_audio.py --continuous --duration 5 --output recordings
```

### 3. Test with CNN Model

Test a single audio file:
```bash
python test_live.py --model ../model/saved_models/cryingsense_cnn_best.pth --audio recordings/recording_20240101_120000.wav
```

Test all recordings in a directory:
```bash
python test_live.py --model ../model/saved_models/cryingsense_cnn_best.pth --audio recordings/
```

## Usage Examples

### Record and Test Workflow

```bash
# Step 1: Record audio
python record_audio.py --duration 5 --output recordings

# Step 2: Test with model
python test_live.py --model ../model/saved_models/cryingsense_cnn_best.pth --audio recordings/ --threshold 0.7
```

### Continuous Testing Loop

```bash
# Terminal 1: Continuous recording
python record_audio.py --continuous --duration 5 --output recordings

# Terminal 2: Watch and test new recordings
# (manually run test_live.py on new files)
```

## Command Line Options

### record_audio.py

```
--duration SECONDS    Recording duration in seconds (default: 5.0)
--output DIR          Output directory for recordings (default: recordings)
--continuous          Enable continuous recording mode
--segments N          Number of segments for continuous mode
--list-devices        List available audio devices
--device INDEX        Specific device index to use
--sample-rate HZ      Sample rate in Hz (default: 16000)
```

### test_live.py

```
--model PATH          Path to trained model checkpoint
--audio PATH          Path to audio file or directory
--threshold FLOAT     Confidence threshold (default: 0.6)
--num-classes N       Number of classes in model (default: 5)
--output FILE         Output JSON file for results
```

## System Requirements

- **Audio Input**: System microphone or audio input device
- **Python**: 3.8+
- **Dependencies**: PyAudio, NumPy, Librosa, PyTorch

### Install PyAudio

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

## Output Format

### Prediction Results

```
==============================================================================
PREDICTION RESULTS
==============================================================================
Audio File: recording_20240101_120000.wav
Timestamp: 2024-01-01T12:00:00.123456

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

## Confidence Threshold

The confidence threshold determines when predictions are considered reliable:

- **≥ threshold**: Prediction is CONFIDENT and can be used
- **< threshold**: Prediction is UNCERTAIN (may be noise or unclear audio)

Default threshold: 0.6 (60%)

Adjust based on your use case:
- **Higher threshold (0.7-0.8)**: Fewer false positives, more rejections
- **Lower threshold (0.5-0.6)**: More predictions, may include false positives

## Troubleshooting

### No Audio Devices Found

- Check that your microphone is connected and enabled
- Run `python record_audio.py --list-devices` to see available devices
- On Linux, ensure your user has audio permissions

### PyAudio Installation Issues

- Install system audio libraries first (see System Requirements)
- Try: `pip install --upgrade pyaudio`
- Alternative: Use `sounddevice` library instead

### Model Not Found

- Train the model first: `cd ../model/training && python train.py`
- Or provide correct model path: `--model path/to/your/model.pth`

### Low Prediction Confidence

- Ensure audio is clear and loud enough
- Check that audio matches training data characteristics
- Try recording closer to the sound source
- Verify sample rate matches training (16kHz)

## Notes

- Audio is recorded in **mono** at **16kHz** sample rate to match model training
- Each recording is **5 seconds** by default (configurable)
- Features extracted: **MFCC**, **Mel Spectrogram**, **Chroma**
- Model input shape: **(batch, 4, 128, time_steps)**
- Inference time: **~20-50ms** on CPU, **~5-10ms** on GPU

## Future Enhancements

- [ ] Real-time streaming inference (no file I/O)
- [ ] Audio visualization during recording
- [ ] Automatic cry detection (start recording on sound)
- [ ] Web interface for testing
- [ ] Mobile app integration

## License

Part of the CryingSense project. See main repository LICENSE for details.
