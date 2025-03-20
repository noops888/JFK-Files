#!/bin/bash
# Installation script for JFK Files PDF Extraction Tool

# Check if running on macOS or Linux
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS installation
    echo "Detected macOS system"
    echo "Installing system dependencies with Homebrew..."
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "Homebrew not found. Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    # Install poppler and tesseract
    brew install poppler tesseract
    
    # Install Python dependencies
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
    
    echo "Installation complete for macOS!"
    
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux installation
    echo "Detected Linux system"
    echo "Installing system dependencies..."
    
    # Install poppler-utils and tesseract-ocr
    sudo apt-get update
    sudo apt-get install -y poppler-utils tesseract-ocr
    
    # Install Python dependencies
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
    
    echo "Installation complete for Linux!"
    
else
    echo "Unsupported operating system: $OSTYPE"
    echo "This script supports macOS and Linux only."
    echo "Please install dependencies manually according to the README.md file."
    exit 1
fi

# Make the extraction script executable
chmod +x extract_embedded_text.py

echo "JFK Files PDF Extraction Tool is now ready to use!"
echo "Run './extract_embedded_text.py --help' for usage instructions."
