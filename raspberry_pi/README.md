# CryingSense — Raspberry Pi 3B+ Pipeline

Self-contained inference pipeline for cry detection on Raspberry Pi 3B+ (1 GB RAM, ARM Cortex-A53 @ 1.4 GHz).

## Folder Structure

```
raspberry_pi/
├── config.py                # Central configuration (audio, model, DB, network)
├── audio_preprocessor.py    # Normalize / trim / resample incoming .wav
├── feature_extractor.py     # Mel-spectrogram-based 4-channel feature builder
├── model.py                 # ONNX model runtime (single inference entry)
├── predictor.py             # Run inference, filter noise, build result dict
├── database_handler.py      # Firebase handler (audio_files, audio_sessions, cry_classifications, device_registrations)
├── app_notifier.py          # Push predictions to Android app (WebSocket / HTTP)
├── recording_trigger.py     # Accept record-start signals from the Android app
├── pipeline.py              # Main orchestrator — ties everything together
├── requirements.txt         # Pip dependencies (RPi-friendly)
├── setup.sh                 # One-shot install + service setup for Raspberry Pi OS
└── saved_models/            # Drop your exported model here
    └── README.md
```

## Quick Start

```bash
# 1. Copy this folder to the Raspberry Pi
scp -r raspberry_pi/ pi@<rpi-ip>:~/cryingsense/

# 2. SSH into the Pi and run setup
ssh pi@<rpi-ip>
cd ~/cryingsense/raspberry_pi
chmod +x setup.sh
./setup.sh            # installs venv, deps, systemd service

# 3. Place your exported model
#    Copy an ONNX (.onnx) model into saved_models/

# 4. Create .env with your Firebase credentials (see config.py for all env vars)
echo 'FIREBASE_CREDENTIALS_PATH=/home/pi/cryingsense/firebase-service-account.json' > .env

# 5. Run
python pipeline.py            # foreground
# or
sudo systemctl start cryingsense   # background (after setup.sh)
```

## Hardware Requirements

| Spec | Detail |
|------|--------|
| Board | Raspberry Pi 3B+ |
| SoC | BCM2837B0, Cortex-A53 (ARMv8) 64-bit @ 1.4 GHz |
| RAM | 1 GB LPDDR2 |
| Storage | 32 GB Micro SD |
| Network | 802.11 b/g/n/ac Wi-Fi, Bluetooth 4.2 |
| Power | 5 V / 2.5 A Micro-USB |
| Mic | USB microphone or I2S MEMS mic via GPIO |

## Android App Integration

The pipeline exposes a lightweight HTTP + WebSocket API on port **8765** (configurable).

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/status` | GET | Health-check / device info |
| `/ws` | WebSocket | Real-time prediction stream |
| `/record` | POST | Trigger on-demand recording |

The Android developer can adjust `APP_API_PORT` and `APP_API_HOST` in `config.py` or via environment variables.

## Firebase Notes

Set these environment variables in `.env` when persistence is enabled:

- `FIREBASE_CREDENTIALS_PATH` (service account JSON file path)
- `FIREBASE_PROJECT_ID` (optional override)
- `FIREBASE_STORAGE_BUCKET` (optional; enables WAV upload to Cloud Storage)
