# CryingSense Trail Run

Real-time infant cry detection and classification system using ESP32 + CNN + MongoDB.

## Overview

This system:
1. **Receives audio** from ESP32 via WiFi (UDP) or USB Serial
2. **Displays real-time waveform** visualization
3. **Ignores** speech and noise classes
4. **Detects** when baby starts crying
5. **Records** the cry audio
6. **Classifies** the cry type using trained CNN model
7. **Displays** classification with probability
8. **Saves** to MongoDB Atlas:
   - Audio files
   - Cry classifications
   - Device registration
   - Session tracking

## Quick Start

### 1. Install Dependencies

```powershell
cd trail_run
pip install -r requirements.txt
```

### 2. Flash ESP32 Firmware

1. Open `esp32_wifi_firmware/` in PlatformIO
2. Edit `src/main.cpp`:
   - Set `WIFI_SSID` to your WiFi name
   - Set `WIFI_PASSWORD` to your WiFi password
   - Set `SERVER_IP` to your computer's IP address
3. Upload to ESP32

### 3. Run the System

```powershell
# Serial mode on COM3 (default)
python main.py

# Serial mode (USB connection)
python main.py --serial COM3

# WiFi mode (opt-in)
python main.py --wifi

# Computer microphone mode (for testing without ESP32)
python main.py --microphone

# Headless mode (no display)
python main.py --headless

# Without database
python main.py --no-db
```

## System Architecture

```
           ┌─────────────┐
           │   ESP32 +   │    UDP/WiFi
           │  INMP441    │ ────────────┐
           └─────────────┘             │
                                       │
           ┌─────────────┐             │
           │   ESP32     │    Serial   │     ┌──────────────────┐
           │   USB       │ ────────────┼────>│  Audio Receiver  │
           └─────────────┘             │     │  (WiFi/Serial/   │
                                       │     │   Microphone)    │
           ┌─────────────┐             │     └────────┬─────────┘
           │  Computer   │  Microphone │              │
           │  Microphone │ ────────────┘              │
           └─────────────┘                            v
                                           ┌──────────────────┐
                                           │   Audio Buffer   │
                                           │  (Circular, 10s) │
                                           └────────┬─────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                      │
                    v                      v                      v
           ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
           │   Waveform   │      │  Classifier  │      │   Display    │
           │   Display    │      │  (CNN Model) │      │   Status     │
           └──────────────┘      └──────┬───────┘      └──────────────┘
                                        │
                                        │ Cry Detected
                                        v
                               ┌──────────────────┐
                               │  Database Handler│
                               │  (MongoDB Atlas) │
                               └──────────────────┘
                                        │
                    ┌───────────────────┬┴─────────────────────┐
                    v                   v                      v
           ┌──────────────┐   ┌──────────────────┐   ┌──────────────────┐
           │ audio_files  │   │cry_classifications│  │audio_sessions   │
           └──────────────┘   └──────────────────┘   └──────────────────┘
```

## File Structure

```
trail_run/
├── main.py                 # Main entry point
├── config.py               # Configuration settings
├── audio_buffer.py         # Thread-safe circular buffer
├── wifi_receiver.py        # WiFi/Serial audio receiver
├── classifier.py           # CNN classifier
├── database_handler.py     # MongoDB operations
├── waveform_display.py     # Real-time visualization
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── recordings/             # Local audio recordings
│   ├── belly_pain/
│   ├── burp/
│   ├── discomfort/
│   ├── hunger/
│   └── tired/
└── esp32_wifi_firmware/    # ESP32 firmware
    ├── platformio.ini
    └── src/
        └── main.cpp
```

## Configuration

Edit `config.py` to customize:

```python
# Model
NUM_CLASSES = 6                   # Must match trained model
CLASS_NAMES = ['belly_pain', 'burp', 'discomfort', 'hunger', 'noise', 'tired']

# Network
WIFI_HOST = "0.0.0.0"             # Listen on all interfaces
WIFI_PORT = 8888                  # UDP port

# Detection
CONFIDENCE_THRESHOLD = 0.70       # Minimum confidence
AMPLITUDE_THRESHOLD = 500         # Cry detection threshold
CRY_DETECTION_CONSECUTIVE = 3     # Consecutive windows needed

# Database
MONGO_URI = "mongodb+srv://..."   # MongoDB connection string

# Classes to ignore (won't be saved to database)
IGNORE_CLASSES = ['noise']        # Only noise is ignored
```

## ESP32 Hardware Setup

### Components
- ESP32 DevKit
- INMP441 I2S Microphone

### Wiring

| INMP441 | ESP32 |
|---------|-------|
| VDD     | 3.3V  |
| GND     | GND   |
| SD      | D16   |
| WS      | D17   |
| SCK     | D18   |
| L/R     | GND   |

### Noise Reduction Tips
1. Use stable 3.3V power (not USB 5V via regulator)
2. Add 10µF capacitor between VDD and GND on INMP441
3. Keep I2S wires short and twisted together
4. Use common ground between ESP32 and microphone

## MongoDB Collections

### cry_classifications
```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "device_id": "abc123",
  "session_id": "uuid",
  "classification": {
    "predicted_class": "hunger",
    "confidence_score": 0.92,
    "all_probabilities": {
      "hunger": 0.92,
      "tired": 0.05,
      "discomfort": 0.02,
      "belly_pain": 0.01,
      "burp": 0.00
    }
  },
  "audio_file_id": "ObjectId"
}
```

### audio_files
```json
{
  "filename": "hunger_20240101_120000.wav",
  "device_id": "abc123",
  "session_id": "uuid",
  "file_data": "<binary wav data>",
  "audio_metadata": {
    "sample_rate": 16000,
    "duration_seconds": 5.2,
    "format": "wav"
  }
}
```

### audio_sessions
```json
{
  "session_id": "uuid",
  "device_id": "abc123",
  "start_time": "2024-01-01T12:00:00Z",
  "end_time": "2024-01-01T14:30:00Z",
  "status": "completed",
  "classification_count": 15,
  "audio_file_count": 15
}
```

### device_registrations
```json
{
  "device_id": "abc123",
  "device_type": "esp32",
  "mac_address": "AA:BB:CC:DD:EE:FF",
  "firmware_version": "1.0.0",
  "first_seen": "2024-01-01T10:00:00Z",
  "last_seen": "2024-01-01T14:30:00Z"
}
```

## Troubleshooting

### No audio received
1. Check ESP32 appears as `COM3` in Device Manager (or run with `--serial <PORT>`)
2. Verify ESP32 firmware is sending binary packets on Serial (USB)
3. Make sure no other app has the COM port open (Arduino Serial Monitor, etc.)
4. For WiFi testing instead, run `python main.py --wifi`
5. Use `--microphone` mode to test with computer mic (no ESP32 needed)

### Microphone issues
1. Check microphone is working in Windows sound settings
2. Run `python -c "import sounddevice; print(sounddevice.query_devices())"` to list devices
3. Make sure sounddevice is installed: `pip install sounddevice`
4. Try adjusting microphone volume/boost in Windows settings
5. Check microphone permissions in Windows Privacy settings

### Model not loading

**Error: "size mismatch" or "Unexpected key(s) in state_dict"**

The trained model must have **6 classes**: `belly_pain`, `burp`, `discomfort`, `hunger`, `noise`, `tired`

If you see dimension errors:
1. Ensure model exists at `model/saved_models/cryingsense_cnn_best.pth`
2. Verify it was trained with 6 classes (NOT 7)
3. If model is from old training with different classes, retrain:
   ```
   python model/training/train.py
   ```
4. Model architecture must match: CryingSenseCNN with 4 input channels

**Note**: The system ignores the `noise` class and only saves the 5 cry types to the database.

### Database connection failed
1. Check MongoDB Atlas credentials in config.py
2. Verify your IP is whitelisted in Atlas Network Access
3. Use `--no-db` flag to run without database

### Poor classification accuracy
1. Ensure proper microphone placement (close to baby)
2. Adjust AMPLITUDE_THRESHOLD if too sensitive/insensitive
3. Re-train model with more data if needed

## Command Line Options

```
usage: main.py [-h] [--serial SERIAL] [--headless] [--no-db] [--port PORT] [--debug]

options:
  -h, --help            Show help message
  --serial, -s SERIAL   Serial port (e.g., COM3, /dev/ttyUSB0)
  --headless, -H        Run without graphical display
  --no-db, -n           Disable database connection
  --port, -p PORT       UDP port for WiFi mode (default: 8888)
  --debug, -d           Enable debug logging
```

## Performance

- **Latency**: ~100ms from cry to classification
- **Accuracy**: Depends on trained model (typically 85-95%)
- **Memory**: ~200MB RAM
- **Network**: ~32KB/s audio data

## License

MIT License - CryingSense Project
