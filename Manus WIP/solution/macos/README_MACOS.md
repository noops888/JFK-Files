# JFK Files PDF Extraction Solution - macOS Optimized

This solution provides a robust method for extracting text from the JFK Files PDF documents without requiring the Gemini API. It is specifically optimized for macOS with enhanced extraction methods and native macOS tools.

## The Problem

The JFK Files collection consists of thousands of PDF documents that contain valuable historical information. While these PDFs already contain embedded OCR text, standard extraction methods often fail to properly access this text on macOS. Previously, the Gemini API was used as the only reliable method to extract this text, but this approach is not sustainable for processing the entire collection due to cost constraints.

## The Solution

This repository provides a macOS-optimized Python-based solution that:

1. Uses multiple specialized PDF libraries to extract embedded OCR text
2. Leverages macOS-native tools for enhanced extraction
3. Tries different extraction methods and selects the best result
4. Implements a combined approach for challenging PDFs
5. Includes a debug mode for troubleshooting extraction issues
6. Processes files in batches to manage system resources
7. Provides detailed extraction metadata for analysis

## Key Components

- `extract_embedded_text_macos.py`: Main script optimized for macOS
- `install_macos.sh`: macOS-specific installation script
- `requirements_macos.txt`: Required Python dependencies for macOS

## Installation on macOS

```bash
# Clone the repository
git clone https://github.com/noops888/JFK-Files.git
cd JFK-Files/Manus\ WIP/solution/

# Run the macOS installation script
./install_macos.sh
```

The installation script will:
1. Install Homebrew if not already installed
2. Install required system dependencies (poppler, tesseract, qpdf)
3. Install Python dependencies
4. Make the extraction script executable

## Usage

```bash
# Basic usage
./extract_embedded_text_macos.py --input /path/to/pdf/files --output /path/to/output/directory

# Process in smaller batches (recommended for large collections)
./extract_embedded_text_macos.py --input /path/to/pdf/files --output /path/to/output/directory --batch 5

# Enable debug mode for troubleshooting
./extract_embedded_text_macos.py --input /path/to/pdf/files --output /path/to/output/directory --debug
```

## How It Works

The macOS-optimized solution uses a multi-library approach with additional macOS-specific enhancements:

1. **PyMuPDF (fitz)**: Enhanced with multiple extraction methods (text, html, dict, json, etc.)
2. **pdfminer.six**: Optimized with various layout analysis parameters
3. **pdfplumber**: Added for its excellent performance on macOS
4. **poppler-utils (pdftotext)**: Enhanced with multiple command-line options
5. **pikepdf**: Improved with better handling of PDF internals
6. **qpdf**: Added for repairing damaged PDFs
7. **macOS native tools**: Added mdls and textutil for macOS-specific extraction

For each PDF, all methods are tried and the one that extracts the most text is selected. For challenging PDFs, a combined approach merges results from all methods to maximize text extraction.

## Debug Mode

The debug mode provides detailed information about the extraction process:
- Shows which extraction methods are being tried
- Reports the parameters and options being used
- Indicates which method works best for each file
- Helps identify why extraction might be failing

Enable debug mode with the `--debug` flag:
```bash
./extract_embedded_text_macos.py --input /path/to/pdf/files --output /path/to/output/directory --debug
```

## Advantages Over Previous Methods

- **No API costs**: Works entirely locally without requiring paid API services
- **macOS optimized**: Specifically designed for macOS environments
- **Enhanced extraction**: Multiple methods and parameters for challenging PDFs
- **Native tools**: Leverages macOS-specific tools for better results
- **Combined approach**: Merges results from multiple methods when needed
- **Transparency**: Provides metadata about which method worked best for each file
- **Debugging**: Includes troubleshooting mode to identify extraction issues

## Performance

In testing, this macOS-optimized solution successfully extracts text from PDFs that other methods failed to process. The extraction quality is comparable to the Gemini API approach but without the associated costs.

## Troubleshooting

If you encounter issues with text extraction:

1. Run the script with the `--debug` flag to see detailed extraction information
2. Ensure all dependencies are correctly installed
3. Try processing a single file at a time with a small batch size
4. Check if the PDF has text extraction restrictions
5. For severely damaged PDFs, try pre-processing with `qpdf --replace-input input.pdf output.pdf`

## Future Improvements

- Add support for parallel processing to increase throughput
- Implement more sophisticated text cleaning for better quality
- Create a GUI interface for easier use
- Add support for extracting and preserving document structure
