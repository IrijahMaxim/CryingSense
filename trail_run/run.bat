@echo off
REM CryingSense Trail Run - Quick Start Script
REM Run this to start the real-time cry detection system

cd /d "%~dp0"

echo ============================================
echo CryingSense Trail Run System
echo ============================================
echo.

REM Activate virtual environment if exists
if exist "..\venv\Scripts\activate.bat" (
    call ..\venv\Scripts\activate.bat
)

REM Check for model file
if not exist "..\model\saved_models\cryingsense_cnn_best.pth" (
    echo ERROR: Model file not found!
    echo Please train the model first:
    echo   python model/training/train.py
    pause
    exit /b 1
)

echo Starting real-time cry detection...
echo Press Ctrl+C to stop
echo.

python main.py %*

pause
