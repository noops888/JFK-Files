#!/bin/bash
# Installation script for cross-platform PDF extraction solution

echo "Installing dependencies for JFK Files PDF extraction solution..."

# Detect operating system
OS="$(uname -s)"
echo "Detected operating system: $OS"

# Install dependencies based on OS
if [ "$OS" == "Darwin" ]; then
    # macOS
    echo "Installing macOS dependencies..."
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "Homebrew not found. Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    
    # Install system dependencies
    echo "Installing system dependencies with Homebrew..."
    brew install poppler qpdf
    
    # Install Python dependencies
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
    
elif [ "$OS" == "Linux" ]; then
    # Linux
    echo "Installing Linux dependencies..."
    
    # Install system dependencies
    echo "Installing system dependencies..."
    if command -v apt-get &> /dev/null; then
        # Debian/Ubuntu
        sudo apt-get update
        sudo apt-get install -y poppler-utils qpdf
    elif command -v yum &> /dev/null; then
        # RHEL/CentOS/Fedora
        sudo yum install -y poppler-utils qpdf
    elif command -v pacman &> /dev/null; then
        # Arch Linux
        sudo pacman -S poppler qpdf
    else
        echo "Unsupported Linux distribution. Please install poppler-utils and qpdf manually."
    fi
    
    # Install Python dependencies
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
    
else
    # Windows or other OS
    echo "Installing dependencies for $OS..."
    
    # Install Python dependencies
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
    
    echo "Note: For optimal performance on Windows, please install poppler and qpdf manually."
    echo "Poppler: https://github.com/oschwartz10612/poppler-windows/releases/"
    echo "QPDF: https://github.com/qpdf/qpdf/releases"
fi

# Make the extraction script executable
chmod +x extract_pdf_text.py

echo "Installation complete!"
echo "Usage: ./extract_pdf_text.py --input /path/to/pdf/files --output /path/to/output/directory"
