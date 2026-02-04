# ESP32 Microphone Module

This directory contains firmware for the ESP32 microphone component of the CryingSense system.

## Responsibilities
- Capture infant cry audio using onboard or external microphone
- Perform initial audio buffering
- Stream audio data to Raspberry Pi for inference
- Provide status indicators (LED, etc.)

## Hardware Requirements
- ESP32 development board
- INMP441 I2S microphone or similar
- Power supply (USB or battery)

## Setup Instructions

### 1. Install Arduino IDE or PlatformIO
- Download and install Arduino IDE or PlatformIO
- Install ESP32 board support

### 2. Install Required Libraries
- ESP32-audioI2S
- WiFi (built-in)
- HTTPClient (built-in)

### 3. Configuration
- Set WiFi credentials
- Set Raspberry Pi IP address
- Configure audio sampling parameters

## Files
- `microphone_capture.ino` - Main firmware (to be implemented)
- `audio_buffer.cpp/h` - Audio buffering logic (to be implemented)
- `network_client.cpp/h` - Network communication (to be implemented)
- `config.h` - Configuration settings (to be implemented)

## Audio Specifications
- Sample Rate: 16,000 Hz
- Bit Depth: 16-bit
- Channels: Mono
- Buffer Size: 80,000 samples (5 seconds)

## Usage
1. Flash firmware to ESP32
2. Connect to configured WiFi network
3. ESP32 will automatically begin capturing and streaming audio

## Power Consumption
- Active (streaming): ~150mA
- Idle (listening): ~80mA
- Deep sleep: <10mA
