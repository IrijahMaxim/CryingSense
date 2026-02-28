# Raspberry Pi IoT Module

This directory contains code for the Raspberry Pi component of the CryingSense system.

## Responsibilities
- Run PyTorch CNN model for inference
- Receive audio data from ESP32 microphone
- Perform real-time cry classification
- Send predictions to backend server via REST API

## Hardware Requirements
- Raspberry Pi 3B+ or newer
- 32GB SD card
- Network connectivity (WiFi or Ethernet)

## Setup Instructions

### 1. Install System Dependencies
```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev
sudo apt-get install -y libatlas-base-dev
```

### 2. Install Python Dependencies
```bash
pip3 install -r ../../model/requirements.txt
```

### 3. Model Deployment
- Copy trained model from `model/saved_models/cryingsense_cnn_quantized.pth`
- Place model file in this directory
- Ensure model is optimized for edge inference (quantized)

## Files
- `inference_server.py` - Main inference service (to be implemented)
- `audio_receiver.py` - ESP32 audio data handler (to be implemented)
- `config.py` - Configuration settings (to be implemented)

## Usage
```bash
python3 inference_server.py
```

## Performance Targets
- Inference latency: <500ms per 5-second sample
- Memory usage: <512MB
- CPU usage: <80% sustained
