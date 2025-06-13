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

# Default port
PORT=5003

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -p, --port PORT    Port to run the server on (default: 5000)"
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

echo "Starting web server on port $PORT..."
echo "The browser will open automatically in a few seconds."
echo ""
echo "Press Ctrl+C to stop the server."
echo ""

# Run the web application
python web_app.py --port $PORT 