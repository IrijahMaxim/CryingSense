@echo off
REM CryingSense Trail Run - Quick Start Script
cd /d "%~dp0"

REM Activate virtual environment if exists
if exist "..\venv\Scripts\activate.bat" (
    call ..\venv\Scripts\activate.bat
)

REM If arguments provided, run directly
if not "%*"=="" (
    python main.py %*
    pause
    exit /b
)

:menu
cls
color 0A
echo ============================================================
echo   CryingSense Trail Run - Launch Menu
echo ============================================================
echo.
echo   1. Run System (Serial Mode - COM3 Default)
echo   2. Run System (Serial/USB Mode)
echo   3. Run System (Computer Microphone)
echo   4. Run System (Headless - No Display)
echo   5. Run System Tests
echo   6. Install Dependencies
echo   7. Exit
echo.
echo ============================================================
set /p choice="Select option (1-7): "

if "%choice%"=="1" goto wifi
if "%choice%"=="2" goto serial
if "%choice%"=="3" goto microphone
if "%choice%"=="4" goto headless
if "%choice%"=="5" goto test
if "%choice%"=="6" goto install
if "%choice%"=="7" goto end
echo Invalid choice. Please try again.
timeout /t 2 >nul
goto menu

:wifi
cls
echo ============================================
echo   Starting Serial Mode (COM3 Default)
echo ============================================
echo.
echo Press Ctrl+C to stop
echo.
python main.py
echo.
pause
goto menu

:serial
cls
echo ============================================
echo   Serial/USB Mode
echo ============================================
echo.
echo Available COM ports:
python -c "import serial.tools.list_ports; [print(f'  {p.device} - {p.description}') for p in serial.tools.list_ports.comports()]" 2>nul
if errorlevel 1 (
    echo   Unable to list ports ^(pyserial not installed^)
    echo   Common ports: COM3, COM4, COM5
)
echo.
set /p port="Enter COM port (e.g., COM3): "
if "%port%"=="" goto serial
echo.
echo Starting on %port%...
echo Press Ctrl+C to stop
echo.
python main.py --serial %port%
echo.
pause
goto menu

:microphone
cls
echo ============================================
echo   Starting Computer Microphone Mode
echo ============================================
echo.
echo Press Ctrl+C to stop
echo.
python main.py --microphone
echo.
pause
goto menu

:headless
cls
echo ============================================
echo   Starting Headless Mode
echo ============================================
echo.
echo Press Ctrl+C to stop
echo.
python main.py --headless
echo.
pause
goto menu

:test
cls
echo ============================================
echo   Running System Tests
echo ============================================
echo.
python test_system.py
echo.
pause
goto menu

:install
cls
echo ============================================
echo   Installing Dependencies
echo ============================================
echo.
pip install -r requirements.txt
echo.
echo Installation complete!
pause
goto menu

:end
exit
