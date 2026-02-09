 # ESP32 Firmware Flashing Instructions

This guide provides step-by-step instructions for flashing the CryingSense firmware onto an ESP32 device.

## Prerequisites

### Hardware Requirements
- ESP32-WROOM-32 development board
- INMP441 or similar I2S MEMS microphone
- USB cable (for flashing and power)
- Jumper wires for connections

### Software Requirements
- Python 3.7 or higher
- esptool (for flashing)
- MicroPython firmware for ESP32
- ampy or rshell (for file transfer)

## Step 1: Install Required Software

### Install esptool
```bash
pip install esptool
```

### Install ampy (for file transfer)
```bash
pip install adafruit-ampy
```

### Install MicroPython libraries (optional, for development)
```bash
pip install mpremote
```

## Step 2: Download MicroPython Firmware

1. Download the latest MicroPython firmware for ESP32:
   - Visit: https://micropython.org/download/esp32/
   - Download: `esp32-xxxxxxxx.bin` (latest stable release)

2. Save the firmware file to a known location

## Step 3: Connect ESP32 to Computer

1. Connect ESP32 to your computer via USB cable
2. Identify the serial port:
   - **Linux/Mac**: `/dev/ttyUSB0` or `/dev/ttyACM0` or `/dev/cu.usbserial-*`
   - **Windows**: `COM3`, `COM4`, etc.

3. Check the connection:
```bash
# Linux/Mac
ls /dev/tty*

# Windows (in PowerShell)
Get-WMIObject Win32_SerialPort | Select-Object Name,DeviceID
```

## Step 4: Erase Flash Memory

Before flashing new firmware, erase the existing flash:

```bash
esptool.py --port /dev/ttyUSB0 erase_flash
```

Replace `/dev/ttyUSB0` with your actual port.

## Step 5: Flash MicroPython Firmware

Flash the MicroPython firmware:

```bash
esptool.py --chip esp32 --port /dev/ttyUSB0 write_flash -z 0x1000 esp32-xxxxxxxx.bin
```

Wait for the flashing process to complete (~30 seconds).

## Step 6: Verify MicroPython Installation

1. Connect to the ESP32 REPL:
```bash
screen /dev/ttyUSB0 115200
# or
minicom -D /dev/ttyUSB0 -b 115200
```

2. Press `Enter` to see the Python prompt: `>>>`

3. Test basic functionality:
```python
>>> print("Hello from ESP32!")
>>> import sys
>>> sys.platform
'esp32'
```

4. Exit REPL: `Ctrl+A` then `K` (screen) or `Ctrl+A` then `X` (minicom)

## Step 7: Upload CryingSense Firmware Files

### Configure WiFi (Required)
Edit `iot/config/device_config.json` and set your WiFi credentials:
```json
{
  "network": {
    "wifi_ssid": "YOUR_WIFI_SSID",
    "wifi_password": "YOUR_WIFI_PASSWORD",
    "gateway_host": "raspberrypi.local"
  }
}
```

### Upload Files
Upload the firmware files to ESP32:

```bash
# Navigate to the iot/firmware directory
cd iot/firmware

# Upload each Python file
ampy --port /dev/ttyUSB0 put audio_capture.py
ampy --port /dev/ttyUSB0 put feature_extraction.py
ampy --port /dev/ttyUSB0 put main.py

# Upload communication modules
cd ../communication
ampy --port /dev/ttyUSB0 put http_client.py
ampy --port /dev/ttyUSB0 put mqtt_client.py

# Upload config file
cd ../config
ampy --port /dev/ttyUSB0 put device_config.json
```

### Alternative: Upload all at once using rshell
```bash
rshell --port /dev/ttyUSB0
# Inside rshell:
cp -r iot/firmware/* /pyboard/
cp -r iot/communication/* /pyboard/
cp iot/config/device_config.json /pyboard/
exit
```

## Step 8: Hardware Connections

### I2S MEMS Microphone (INMP441) to ESP32

| INMP441 Pin | ESP32 Pin | Function |
|-------------|-----------|----------|
| VDD         | 3.3V      | Power    |
| GND         | GND       | Ground   |
| SD          | GPIO 32   | Serial Data |
| WS          | GPIO 15   | Word Select (LRCK) |
| SCK         | GPIO 14   | Serial Clock |
| L/R         | GND       | Left channel select |

**Note**: Pin configuration can be changed in `device_config.json`

## Step 9: Run CryingSense Firmware

### Manual Start (for testing)
1. Connect to REPL
2. Run:
```python
>>> import main
>>> main.main()
```

### Auto-start on Boot
Create a `boot.py` file that runs on startup:

```bash
# Create boot.py locally
cat > boot.py << 'EOF'
import time
import network

# Connect to WiFi
wlan = network.WLAN(network.STA_IF)
wlan.active(True)
if not wlan.isconnected():
    print('Connecting to WiFi...')
    wlan.connect('YOUR_SSID', 'YOUR_PASSWORD')
    timeout = 10
    while not wlan.isconnected() and timeout > 0:
        time.sleep(1)
        timeout -= 1

if wlan.isconnected():
    print('WiFi connected:', wlan.ifconfig())
    # Start main application
    import main
    main.main()
else:
    print('WiFi connection failed')
EOF

# Upload boot.py
ampy --port /dev/ttyUSB0 put boot.py
```

## Step 10: Verify Operation

1. Reset the ESP32 (press the reset button)
2. Monitor serial output:
```bash
screen /dev/ttyUSB0 115200
```

3. Expected output:
```
CryingSense ESP32 initialized: ESP32_001
Sample rate: 16000 Hz
Capture duration: 5.0 seconds
Starting continuous cry detection...
Listening for cry...
```

## Troubleshooting

### Issue: Cannot connect to serial port
**Solution**: 
- Check USB cable connection
- Install USB-to-Serial drivers (CP210x or CH340)
- Check permissions: `sudo usermod -a -G dialout $USER` (Linux)

### Issue: Flashing fails
**Solution**:
- Hold the `BOOT` button while flashing
- Try lower baud rate: `esptool.py --port /dev/ttyUSB0 --baud 115200 write_flash ...`

### Issue: Files upload fails
**Solution**:
- Ensure MicroPython is properly installed
- Reset ESP32 and try again
- Use smaller file sizes or upload one by one

### Issue: WiFi connection fails
**Solution**:
- Verify SSID and password in configuration
- Check WiFi signal strength
- Ensure 2.4GHz WiFi (ESP32 doesn't support 5GHz)

### Issue: Audio capture fails
**Solution**:
- Verify I2S pin connections
- Check microphone power (3.3V)
- Test microphone with simple audio capture script

## Advanced Configuration

### Change Detection Threshold
Edit `device_config.json`:
```json
{
  "audio": {
    "detection_threshold": 0.15
  }
}
```

### Enable MQTT Instead of HTTP
Edit `device_config.json`:
```json
{
  "communication": {
    "protocol": "mqtt"
  },
  "network": {
    "mqtt_broker": "raspberrypi.local",
    "mqtt_port": 1883
  }
}
```

### Adjust Detection Interval
Edit `device_config.json`:
```json
{
  "operation": {
    "detection_interval": 5.0
  }
}
```

## Updating Firmware

To update firmware files:
```bash
# Remove old file
ampy --port /dev/ttyUSB0 rm main.py

# Upload new file
ampy --port /dev/ttyUSB0 put main.py

# Reset ESP32
```

## Monitoring and Debugging

### View Logs
```bash
screen /dev/ttyUSB0 115200
```

### Remote REPL via WebREPL (optional)
1. Enable WebREPL in MicroPython:
```python
>>> import webrepl_setup
```

2. Access via browser: `http://micropython.org/webrepl/`

## Support

For issues or questions:
- Check ESP32 documentation: https://docs.espressif.com/
- MicroPython ESP32 docs: https://docs.micropython.org/en/latest/esp32/
- Project repository: [GitHub link]

## Safety Notes

- Do not connect ESP32 to power > 3.3V
- Avoid static discharge
- Use proper USB power supply (min 500mA)
- Keep device away from water and excessive heat
