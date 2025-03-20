# Cross-Platform PDF Text Extraction Solution for JFK Files

This solution provides a robust method for extracting text from the JFK Files PDF documents without requiring the Gemini API. It works across platforms (macOS, Linux, Windows) while prioritizing macOS-native tools when available for optimal extraction.

## The Problem

The JFK Files collection consists of thousands of PDF documents that contain valuable historical information. While these PDFs already contain embedded OCR text, standard extraction methods often fail to properly access this text. Previously, the Gemini API was used as the only reliable method to extract this text, but this approach is not sustainable for processing the entire collection due to cost constraints.

## The Solution

This cross-platform solution:

1. Prioritizes macOS-native tools when available (mdls, textutil) which have proven most effective
2. Provides robust fallback methods for non-macOS platforms
3. Uses multiple specialized PDF libraries to extract embedded OCR text
4. Tries different extraction methods and selects the best result
5. Implements a combined approach for challenging PDFs
6. Processes files in batches to manage system resources
7. Provides detailed extraction metadata for analysis

## Key Components

- `extract_pdf_text.py`: Main cross-platform extraction script
- `install.sh`: Installation script for dependencies
- `requirements.txt`: Required Python dependencies

## Installation

```bash
# Clone the repository
git clone https://github.com/noops888/JFK-Files.git
cd JFK-Files/Manus\ WIP/solution/cross_platform

# Run the installation script
./install.sh
```

The installation script will:
1. Detect your operating system
2. Install required system dependencies based on your OS
3. Install Python dependencies
4. Make the extraction script executable

## Usage

```bash
# Basic usage
./extract_pdf_text.py --input /path/to/pdf/files --output /path/to/output/directory

# Process in smaller batches (recommended for large collections)
./extract_pdf_text.py --input /path/to/pdf/files --output /path/to/output/directory --batch 5

# Enable debug mode for troubleshooting
./extract_pdf_text.py --input /path/to/pdf/files --output /path/to/output/directory --debug
```

## How It Works

The solution uses a multi-library approach with platform-specific optimizations:

1. **PyMuPDF (fitz)**: Enhanced with multiple extraction methods (text, html, dict, json, etc.)
2. **pdfminer.six**: Optimized with various layout analysis parameters
3. **pdfplumber**: Added for its excellent performance with structured content
4. **poppler-utils (pdftotext)**: Enhanced with multiple command-line options
5. **pikepdf**: Improved with better handling of PDF internals
6. **qpdf**: Added for repairing damaged PDFs
7. **macOS native tools**: Added mdls and textutil for macOS-specific extraction (macOS only)

For each PDF, all applicable methods are tried and the one that extracts the most text is selected. For challenging PDFs, a combined approach merges results from all methods to maximize text extraction.

## Platform-Specific Optimizations

### macOS
- Uses native macOS tools (mdls, textutil) which have proven most effective
- Falls back to cross-platform methods if native tools fail

### Linux
- Uses poppler-utils and qpdf for optimal extraction
- Implements multiple extraction methods with various parameters

### Windows
- Provides guidance for installing necessary dependencies
- Uses Python-based extraction libraries that work well on Windows

## Advantages Over Previous Methods

- **No API costs**: Works entirely locally without requiring paid API services
- **Cross-platform**: Works on macOS, Linux, and Windows
- **macOS optimized**: Prioritizes macOS-native tools when available
- **Enhanced extraction**: Multiple methods and parameters for challenging PDFs
- **Combined approach**: Merges results from multiple methods when needed
- **Transparency**: Provides metadata about which method worked best for each file

## Performance

In testing, this solution successfully extracts text from PDFs that other methods failed to process. On macOS, the extraction quality is comparable to the Gemini API approach but without the associated costs. On other platforms, the solution provides the best possible extraction using available tools.

## Troubleshooting

If you encounter issues with text extraction:

1. Run the script with the `--debug` flag to see detailed extraction information
2. Ensure all dependencies are correctly installed
3. Try processing a single file at a time with a small batch size
4. Check if the PDF has text extraction restrictions
5. For severely damaged PDFs, try pre-processing with `qpdf --replace-input input.pdf`
