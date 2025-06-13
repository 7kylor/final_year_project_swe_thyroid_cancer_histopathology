#!/bin/bash

# Thyroid Cancer Histopathology Web Interface Launcher

echo "============================================"
echo "Thyroid Cancer Histopathology Analysis"
echo "Web-Based Deployment Interface"
echo "============================================"
echo ""

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Check if required packages are installed
echo "Checking dependencies..."
python -c "import torch" 2>/dev/null || { echo "PyTorch not installed. Please run: pip install -r requirements.txt"; exit 1; }
python -c "import flask" 2>/dev/null || { echo "Flask not installed. Please run: pip install -r requirements.txt"; exit 1; }

# Function to check if port is available
check_port() {
    local port=$1
    if command -v lsof &> /dev/null; then
        lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1
    elif command -v netstat &> /dev/null; then
        netstat -an | grep ":$port " | grep LISTEN >/dev/null 2>&1
    elif command -v ss &> /dev/null; then
        ss -ln | grep ":$port " >/dev/null 2>&1
    else
        # Fallback: try to bind to the port
        (exec 3<>/dev/tcp/127.0.0.1/$port) 2>/dev/null && exec 3<&- && exec 3>&-
    fi
}

# Function to find available port
find_available_port() {
    local start_port=$1
    local max_port=$((start_port + 50))  # Try up to 50 ports
    
    for ((port=start_port; port<=max_port; port++)); do
        if ! check_port $port; then
            echo $port
            return 0
        fi
    done
    
    echo "Could not find available port in range $start_port-$max_port" >&2
    return 1
}

# Default port
DEFAULT_PORT=5000
PORT=$DEFAULT_PORT
USER_SPECIFIED_PORT=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--port)
            PORT="$2"
            USER_SPECIFIED_PORT=true
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -p, --port PORT    Port to run the server on (default: auto-detect starting from 5000)"
            echo "  -h, --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Check if port is available and find alternative if needed
if check_port $PORT; then
    if [ "$USER_SPECIFIED_PORT" = true ]; then
        echo "Port $PORT is in use, finding alternative..."
        AVAILABLE_PORT=$(find_available_port $PORT)
    else
        echo "Port $PORT is in use, finding alternative..."
        AVAILABLE_PORT=$(find_available_port $DEFAULT_PORT)
    fi
    
    if [ $? -eq 0 ]; then
        PORT=$AVAILABLE_PORT
        echo "Using port $PORT instead."
    else
        echo "Error: Could not find an available port."
        exit 1
    fi
fi



echo "Starting web server on port $PORT..."
echo "The browser will open automatically in a few seconds."
echo ""
echo "Press Ctrl+C to stop the server."
echo ""

# Run the web application
python web_app.py --port $PORT 