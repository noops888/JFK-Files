#!/bin/bash
# Installation script for JFK Files PDF Extraction Tool - macOS Optimized Version

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "Error: This script is specifically optimized for macOS."
    echo "Please use the standard install.sh script for Linux systems."
    exit 1
fi

echo "Installing JFK Files PDF Extraction Tool - macOS Optimized Version"

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install required system dependencies
echo "Installing system dependencies with Homebrew..."
brew install poppler tesseract qpdf

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements_macos.txt

# Make the extraction script executable
chmod +x extract_embedded_text_macos.py

echo "Installation complete!"
echo "Run './extract_embedded_text_macos.py --help' for usage instructions."
echo ""
echo "For best results on challenging PDFs, use the debug mode:"
echo "./extract_embedded_text_macos.py --input /path/to/pdfs --output /path/to/output --debug"
