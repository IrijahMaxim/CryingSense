# CryingSense Audio Test Tools

This folder contains tools for testing audio input, recording samples, and running live inference.

---

## Quick Start

```bash
# Activate virtual environment first
cd "P:\VScode Lobby\CryingSense"
.\venv\Scripts\Activate.ps1

# Navigate to audio_test folder
cd audio_test
```

---

## Tools Overview

| Tool | Purpose | Requires Model |
|------|---------|----------------|
| `tester.py` | Test microphone & visualize audio | No |
| `record_audio.py` | Record audio clips | No |
| `test_live.py` | Run inference on audio file | Yes |
| `live.py` | Continuous live monitoring | Yes |
| `sound_level_meter.py` | Real-time level meter | No |

---

## tester.py - Audio Input Tester

Test your microphone and audio quality without needing a trained model.

### Features
- Real-time dB level monitoring
- 6-band frequency spectrum analyzer
- Clipping and silence detection
- Audio quality assessment

### Usage

```bash
# Live audio testing (continuous)
python tester.py

# Record a test clip (5 seconds default)
python tester.py --record

# Record for 10 seconds
python tester.py --record --duration 10

# Use specific audio device
python tester.py --device 1

# List available audio devices
python tester.py --list-devices
```

### Options
| Option | Description | Default |
|--------|-------------|---------|
| `--sample-rate` | Audio sample rate (Hz) | 16000 |
| `--device` | Audio device index | System default |
| `--record` | Record a test clip | False |
| `--duration` | Recording duration (seconds) | 5.0 |
| `--output` | Output directory for recordings | `recordings` |
| `--list-devices` | List audio devices and exit | - |

---

## record_audio.py - Audio Recorder

Record audio clips for dataset collection or testing.

### Features
- Real-time sound level display during recording
- Countdown timer
- Saves as WAV format (16kHz, 16-bit mono)

### Usage

```bash
# Record 5 seconds (default)
python record_audio.py

# Record for 10 seconds
python record_audio.py --duration 10

# Specify output filename
python record_audio.py --output my_recording.wav

# Disable real-time level display
python record_audio.py --no-levels

# Use specific audio device
python record_audio.py --device 1
```

### Options
| Option | Description | Default |
|--------|-------------|---------|
| `--duration` | Recording duration (seconds) | 5.0 |
| `--output` | Output filename | `recording_YYYYMMDD_HHMMSS.wav` |
| `--sample-rate` | Audio sample rate (Hz) | 16000 |
| `--device` | Audio device index | System default |
| `--no-levels` | Disable real-time level display | False |

---

## test_live.py - Single File Inference

Run cry classification on a single audio file.

### Features
- Load and analyze WAV audio files
- Display prediction with confidence scores
- Show audio level statistics
- Feature extraction timing info

### Usage

```bash
# Basic inference
python test_live.py --audio recordings/my_audio.wav --model ../model/saved_models/cryingsense_cnn_best.pth

# Use 6-class model
python test_live.py --audio recordings/my_audio.wav --model ../model/saved_models/model_6class.pth --num-classes 6
```

### Options
| Option | Description | Default |
|--------|-------------|---------|
| `--audio` | Path to audio file (required) | - |
| `--model` | Path to model checkpoint | `../model/saved_models/cryingsense_cnn_best.pth` |
| `--num-classes` | Number of classes | 5 |

### Output Example
```
Audio Level: [████████████████████···················] -28.1 dB (NORMAL)

Prediction: DISCOMFORT
Confidence: 85.3%
Status: HIGH CONFIDENCE

Class Probabilities:
  discomfort   [██████████████████████████████] 85.3% <--
  hunger       [████······························] 8.2%
  tired        [██································] 3.5%
  belly_pain   [█·································] 2.1%
  burp         [·································· ] 0.9%
```

---

## live.py - Continuous Live Monitoring

Run continuous real-time inference on microphone input.

### Features
- Non-stop audio monitoring
- Rolling 5-second buffer for inference
- Real-time predictions every 1 second
- Alert system for high-confidence detections
- Optional saving of detected audio clips

### Usage

```bash
# Start live monitoring
python live.py --model ../model/saved_models/cryingsense_cnn_best.pth

# Save detected cries
python live.py --model ../model/saved_models/cryingsense_cnn_best.pth --save

# Custom confidence threshold (80%)
python live.py --model ../model/saved_models/cryingsense_cnn_best.pth --threshold 0.8

# Use specific audio device
python live.py --model ../model/saved_models/cryingsense_cnn_best.pth --device 1

# List available audio devices
python live.py --list-devices
```

### Options
| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Path to model checkpoint | `../model/saved_models/cryingsense_cnn_best.pth` |
| `--num-classes` | Number of classes | 5 |
| `--threshold` | Confidence threshold for alerts | 0.6 |
| `--duration` | Audio chunk duration (seconds) | 5.0 |
| `--sample-rate` | Audio sample rate (Hz) | 16000 |
| `--device` | Audio device index | System default |
| `--save` | Save audio when cry detected | False |
| `--output` | Output directory for detections | `detections` |
| `--list-devices` | List audio devices and exit | - |

### Display
```
======================================================================
  CRYINGSENSE LIVE MONITOR
======================================================================

  AUDIO INPUT
  ------------------------------------------------------------------
  Level: [███████████████████████·························] -25.3 dB
  Peak:  -18.2 dB  |  Status: [■] NORMAL

  Buffer: [████████████████████] 100.0%  (80000/80000 samples)

  PREDICTION
  ------------------------------------------------------------------
  >>> DETECTED: HUNGER (78.5%) <<<
  ALERT STATUS: HIGH CONFIDENCE DETECTION!

  Class Probabilities:
    hunger       [████████████████████████████··] 78.5% <--
    discomfort   [█████·························] 12.3%
    ...

  ------------------------------------------------------------------
  Alerts: 3  |  Threshold: 60%  |  Time: 14:32:15

======================================================================
  Press Ctrl+C to stop
======================================================================
```

---

## sound_level_meter.py - Level Meter Utility

Standalone sound level meter with spectrum analyzer.

### Features
- Real-time dB level with visual bar
- 16-band frequency spectrum
- Peak hold indicator
- Audio status classification

### Usage

```bash
# Run standalone level meter
python sound_level_meter.py

# Use specific audio device
python sound_level_meter.py --device 1

# List audio devices
python sound_level_meter.py --list-devices
```

### Exported Functions
Other scripts can import utilities from this module:

```python
from sound_level_meter import calculate_db, create_level_bar, get_level_status

# Calculate dB from audio samples
db = calculate_db(audio_chunk)

# Create visual level bar
bar = create_level_bar(db, width=40)

# Get status text
status, symbol = get_level_status(db)
```

---

## Audio Device Selection

If you have multiple microphones or audio interfaces:

```bash
# List all available input devices
python tester.py --list-devices

# Output:
# Available Audio Input Devices
# ==============================================================
#   Device 0: Microphone (Realtek) [DEFAULT]
#             Sample Rate: 44100 Hz, Channels: 2
#   Device 1: USB Audio Device
#             Sample Rate: 48000 Hz, Channels: 1
#   Device 2: Webcam Microphone
#             Sample Rate: 16000 Hz, Channels: 1
# ==============================================================

# Use device 1
python tester.py --device 1
python live.py --device 1 --model ../model/saved_models/cryingsense_cnn_best.pth
```

---

## Troubleshooting

### No audio input detected
1. Check microphone is connected and enabled
2. Run `python tester.py --list-devices` to see available devices
3. Try specifying device explicitly: `--device 0`

### Clipping warnings
- Lower your microphone input gain
- Move microphone further from sound source

### Very low levels
- Increase microphone gain
- Check microphone isn't muted
- Speak closer to microphone

### Model loading errors
- Ensure model file exists at specified path
- Check `--num-classes` matches model architecture (5 or 6)

### PyAudio errors on Windows
```bash
pip install pyaudio
# If that fails:
pip install pipwin
pipwin install pyaudio
```

---

## Class Labels

The model classifies cries into these categories:

| Class | Description |
|-------|-------------|
| `belly_pain` | Cry indicating stomach discomfort |
| `burp` | Burping/gas-related sounds |
| `discomfort` | General discomfort cry |
| `hunger` | Hungry cry pattern |
| `tired` | Sleepy/tired cry |
| `noise` | Non-cry sounds (6-class model only) |

---

## File Structure

```
audio_test/
├── README.md           # This file
├── __init__.py
├── live.py             # Continuous live monitoring
├── record_audio.py     # Audio recording
├── sound_level_meter.py # Level meter utility
├── test_live.py        # Single file inference
├── tester.py           # Audio input testing
├── recordings/         # Saved recordings
└── detections/         # Saved cry detections (live.py --save)
```

---

## System Requirements

- **Audio Input**: System microphone or audio input device
- **Python**: 3.8+
- **Dependencies**: PyAudio, NumPy, Librosa, PyTorch

## License

Part of the CryingSense project. See main repository LICENSE for details.
