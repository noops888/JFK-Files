# JFK Files PDF Extraction Solution

This repository provides two robust methods for extracting text from the JFK Files PDF documents without requiring the Gemini API:

1. **Linux Version**: Works well on Linux systems and uses multiple specialized PDF libraries
2. **macOS Version**: Specifically optimized for macOS with enhanced extraction methods and native macOS tools

## The Problem

The JFK Files collection consists of thousands of PDF documents that contain valuable historical information. While these PDFs already contain embedded OCR text, standard extraction methods often fail to properly access this text. Previously, the Gemini API was used as the only reliable method to extract this text, but this approach is not sustainable for processing the entire collection due to cost constraints.

## Platform-Specific Solutions

### Linux Solution
- `linux/extract_embedded_text.py`: Main script for Linux systems
- `linux/install.sh`: Linux installation script
- `linux/requirements.txt`: Required Python dependencies for Linux
- [View Linux-specific README](linux/README.md)

### macOS Solution
- `macos/extract_embedded_text_macos.py`: Main script optimized for macOS
- `macos/install_macos.sh`: macOS-specific installation script
- `macos/requirements_macos.txt`: Required Python dependencies for macOS
- [View macOS-specific README](macos/README_MACOS.md)

## Quick Start

### For Linux Users:
```bash
cd linux
./install.sh
./extract_embedded_text.py --input /path/to/pdf/files --output /path/to/output/directory
```

### For macOS Users:
```bash
cd macos
./install_macos.sh
./extract_embedded_text_macos.py --input /path/to/pdf/files --output /path/to/output/directory
```

## How It Works

Both solutions use a multi-library approach to maximize text extraction success, but with platform-specific optimizations:

### Common Features
- Multiple specialized PDF libraries to extract embedded OCR text
- Smart selection of the best extraction method for each PDF
- Batch processing to manage system resources
- Detailed extraction metadata for analysis

### Linux-Specific Features
- Optimized for Linux environments
- Uses pikepdf, PyMuPDF, pdfminer.six, and poppler-utils

### macOS-Specific Features
- Leverages macOS-native tools (mdls, textutil)
- Adds pdfplumber library which works well on macOS
- Enhanced extraction parameters for macOS compatibility
- Implements a combined approach for challenging PDFs
- Includes a debug mode for troubleshooting extraction issues

## Advantages Over Previous Methods

- **No API costs**: Works entirely locally without requiring paid API services
- **Platform-optimized**: Specific solutions for Linux and macOS
- **Speed**: Much faster than OCR-based approaches since it extracts existing text
- **Reliability**: Multiple extraction methods ensure higher success rates
- **Transparency**: Provides metadata about which method worked best for each file
- **Scalability**: Can process the entire collection on a standard computer
