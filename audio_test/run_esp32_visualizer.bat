@echo off
echo ========================================
echo ESP32 INMP441 Audio Visualizer
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7+ from https://www.python.org/
    pause
    exit /b 1
)

echo Checking dependencies...
pip show pyqtgraph >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r esp32_visualizer_requirements.txt
) else (
    echo Dependencies OK
)

echo.
echo Available COM Ports:
python -c "import serial.tools.list_ports; [print(f'  {p.device}: {p.description}') for p in serial.tools.list_ports.comports()]"

echo.
echo Starting visualizer...
echo.
python esp32_audio_visualizer.py %1

pause
