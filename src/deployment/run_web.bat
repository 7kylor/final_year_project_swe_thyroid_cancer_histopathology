@echo off
REM Thyroid Cancer Histopathology Web Interface Launcher for Windows

echo ============================================
echo Thyroid Cancer Histopathology Analysis
echo Web-Based Deployment Interface
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.7 or higher from https://www.python.org
    pause
    exit /b 1
)

REM Check if required packages are installed
echo Checking dependencies...
python -c "import torch" 2>nul
if errorlevel 1 (
    echo PyTorch not installed. Please run: pip install -r requirements.txt
    pause
    exit /b 1
)

python -c "import flask" 2>nul
if errorlevel 1 (
    echo Flask not installed. Please run: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Set default port
set PORT=5001

REM Parse command line arguments
if "%1"=="-p" set PORT=%2
if "%1"=="--port" set PORT=%2
if "%1"=="-h" goto :help
if "%1"=="--help" goto :help
if "%1"=="/?" goto :help

echo Starting web server on port %PORT%...
echo The browser will open automatically in a few seconds.
echo.
echo Press Ctrl+C to stop the server.
echo.

REM Run the web application
python web_app.py --port %PORT%
goto :end

:help
echo Usage: %0 [options]
echo Options:
echo   -p, --port PORT    Port to run the server on (default: 5001)
echo   -h, --help, /?     Show this help message

:end 