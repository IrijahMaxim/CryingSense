# ESP32 INMP441 Audio Visualizer

Real-time audio visualization for ESP32 INMP441 microphone via Serial connection.

## Features

🎵 **Waveform Display**: Shows the amplitude waveform of audio signals in real-time  
📊 **Frequency Spectrum**: FFT analysis displaying frequency content (0-8 kHz)  
📈 **RMS Monitoring**: Real-time RMS (Root Mean Square) amplitude tracking  
👶 **Baby Cry Detection**: Visual feedback for baby crying detection with threshold markers  
💡 **LED Feedback**: Shows current LED brightness based on sound level

## Installation

1. Install Python dependencies:
```bash
pip install -r esp32_visualizer_requirements.txt
```

Or install manually:
```bash
pip install pyserial numpy PyQt5 pyqtgraph
```

2. Upload the ESP32 code to your device (already configured in main.cpp)

## Usage

### Option 1: Auto-detect COM port
```bash
python esp32_audio_visualizer.py
```
Then select your COM port from the dropdown and click "Connect"

### Option 2: Specify COM port
```bash
python esp32_audio_visualizer.py COM3
```
Replace `COM3` with your ESP32's COM port

### Finding Your COM Port

**Windows:**
- Open Device Manager → Ports (COM & LPT)
- Look for "USB-SERIAL CH340" or "CP210x" or similar

**List ports in terminal:**
```bash
python -c "import serial.tools.list_ports; [print(p) for p in serial.tools.list_ports.comports()]"
```

## Display Components

### 1. Audio Waveform (Top)
- Shows 2 seconds of audio data
- Auto-scales to keep signal visible
- Blue line represents amplitude over time

### 2. RMS & Status Display (Middle)
- **RMS**: Current Root Mean Square amplitude
- **Peak**: Maximum RMS value since start
- **Status**: Current detection status with color coding:
  - 🟢 Green: Quiet/Normal
  - 🟠 Orange: Baby crying detected
  - 🔴 Red: Loud crying

### 3. Frequency Spectrum (Bottom)
- FFT analysis showing frequency content
- Red dashed lines: Baby cry frequency range (300-600 Hz)
- Range: 0-8000 Hz (full microphone range)
- Magnitude shown in dB scale

## Baby Cry Detection Thresholds

The visualization shows real-time detection with these thresholds:
- **Ambient**: < 300 (Room noise)
- **Crying**: 800-1500 (Baby crying)  
- **Loud Cry**: > 1500 (Intense crying)

## Troubleshooting

### "No COM ports found!"
- Ensure ESP32 is connected via USB
- Install CH340 or CP210x drivers if needed
- Check Device Manager (Windows) for port issues

### Connection Error
- Close any Serial Monitor or other program using the port
- Try unplugging and replugging the ESP32
- Check baud rate is 115200 (default)

### No waveform displayed
- Ensure `OUTPUT_RAW_SAMPLES` is set to `true` in main.cpp (default)
- Check that data is being sent (should see "SAMPLES:" in raw serial output)
- Verify microphone connections (SD, WS, SCK pins)

### Microphone not detecting sound
- Test with a louder sound source
- Check INMP441 wiring:
  - SD → GPIO 16
  - WS → GPIO 17  
  - SCK → GPIO 18
  - VDD → 3.3V
  - GND → GND

## Configuration

### ESP32 Configuration (main.cpp)
```cpp
#define OUTPUT_RAW_SAMPLES true    // Enable/disable raw sample output
#define SAMPLES_TO_OUTPUT 32       // Samples per transmission (affects bandwidth)
#define AMBIENT_THRESHOLD 300.0    // Adjust for environment noise
#define CRYING_THRESHOLD 800.0     // Baby crying detection threshold
```

### Python Configuration (esp32_audio_visualizer.py)
```python
sample_rate = 16000              # Must match ESP32 sample rate
self.waveform_duration = 2.0     # Seconds of waveform to display
self.fft_size = 1024             # FFT resolution
```

## Performance Notes

- Update rate: 20 FPS (50ms refresh)
- Serial bandwidth: ~115200 baud
- Sample rate: 16 kHz (matches typical baby cry detection)
- Waveform buffer: 2 seconds (32,000 samples)
- FFT resolution: 1024 points (~15.6 Hz per bin)

## Technical Details

### Audio Processing
- Uses I2S interface for high-quality audio capture
- 16-bit samples at 16 kHz sample rate
- RMS calculation for volume estimation
- FFT with Hanning window for frequency analysis

### Data Format
The ESP32 sends mixed text and sample data:
```
RMS: 450.2 | LED: 95 | Status: Normal
SAMPLES:-123,456,-789,234,...
⚠️  Baby crying detected
```

The visualizer parses both formats for comprehensive display.

## License

Part of the CryingSense project - Baby cry detection system
