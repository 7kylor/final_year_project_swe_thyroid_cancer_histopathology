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
set DEFAULT_PORT=5001
set PORT=%DEFAULT_PORT%
set USER_SPECIFIED_PORT=false

REM Parse command line arguments
if "%1"=="-p" (
    set PORT=%2
    set USER_SPECIFIED_PORT=true
)
if "%1"=="--port" (
    set PORT=%2
    set USER_SPECIFIED_PORT=true
)
if "%1"=="-h" goto :help
if "%1"=="--help" goto :help
if "%1"=="/?" goto :help

REM Function to check if port is available (Windows)
echo Checking if port %PORT% is available...
netstat -an | findstr ":%PORT% " | findstr "LISTENING" >nul 2>&1
if %errorlevel% equ 0 (
    echo Port %PORT% is in use, trying alternative ports...
    if "%USER_SPECIFIED_PORT%"=="true" (
        call :find_available_port %PORT%
    ) else (
        call :find_available_port %DEFAULT_PORT%
    )
    if !errorlevel! neq 0 (
        echo Error: Could not find an available port.
        pause
        exit /b 1
    )
) else (
    echo Port %PORT% is available.
)

echo Starting web server on port %PORT%...
echo The browser will open automatically in a few seconds.
echo.
echo Press Ctrl+C to stop the server.
echo.

REM Run the web application
python web_app.py --port %PORT%
goto :end

:find_available_port
setlocal enabledelayedexpansion
set start_port=%1
set /a max_port=%start_port% + 50

for /l %%i in (%start_port%,1,%max_port%) do (
    netstat -an | findstr ":%%i " | findstr "LISTENING" >nul 2>&1
    if !errorlevel! neq 0 (
        set PORT=%%i
        echo Using port %%i instead.
        endlocal & set PORT=%PORT%
        exit /b 0
    )
)

echo Could not find available port in range %start_port%-%max_port%
endlocal
exit /b 1
goto :end

:help
echo Usage: %0 [options]
echo Options:
echo   -p, --port PORT    Port to run the server on (default: auto-detect starting from 5001)
echo   -h, --help, /?     Show this help message

:end 