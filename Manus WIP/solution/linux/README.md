# JFK Files PDF Extraction Solution - Linux Version

This solution provides a robust method for extracting text from the JFK Files PDF documents without requiring the Gemini API. It works well on Linux systems and uses multiple specialized PDF libraries to access the embedded OCR text.

## The Problem

The JFK Files collection consists of thousands of PDF documents that contain valuable historical information. While these PDFs already contain embedded OCR text, standard extraction methods (PyPDF2, pdfplumber, etc.) often fail to properly access this text. Previously, the Gemini API was used as the only reliable method to extract this text, but this approach is not sustainable for processing the entire collection due to cost constraints.

## The Solution

This repository provides a Linux-optimized Python-based solution that:

1. Uses multiple specialized PDF libraries to extract embedded OCR text
2. Tries different extraction methods and selects the best result
3. Processes files in batches to manage system resources
4. Provides detailed extraction metadata for analysis

## Key Components

- `extract_embedded_text.py`: Main script for extracting embedded OCR text from PDFs
- `install.sh`: Linux installation script
- `requirements.txt`: Required Python dependencies

## Installation on Linux

```bash
# Clone the repository
git clone https://github.com/noops888/JFK-Files.git
cd JFK-Files/Manus\ WIP/solution/linux/

# Run the installation script
./install.sh
```

The installation script will:
1. Install required system dependencies (poppler-utils, tesseract-ocr)
2. Install Python dependencies
3. Make the extraction script executable

## Usage

```bash
# Basic usage
./extract_embedded_text.py --input /path/to/pdf/files --output /path/to/output/directory

# Process in smaller batches (recommended for large collections)
./extract_embedded_text.py --input /path/to/pdf/files --output /path/to/output/directory --batch 10
```

## How It Works

The solution uses a multi-library approach to maximize text extraction success:

1. **PyMuPDF (fitz)**: Fast and reliable for most PDFs
2. **pdfminer.six**: Good for complex PDF structures
3. **poppler-utils (pdftotext)**: Command-line utility that works well for simple PDFs
4. **pikepdf**: Low-level access to PDF internals

For each PDF, all methods are tried and the one that extracts the most text is selected. This approach ensures the highest possible success rate across the diverse JFK Files collection.

## Advantages Over Previous Methods

- **No API costs**: Works entirely locally without requiring paid API services
- **Speed**: Much faster than OCR-based approaches since it extracts existing text
- **Reliability**: Multiple extraction methods ensure higher success rates
- **Transparency**: Provides metadata about which method worked best for each file
- **Scalability**: Can process the entire collection on a standard Linux computer

## Performance

In testing, this solution successfully extracted text from PDFs that other methods failed to process. The extraction quality is comparable to the Gemini API approach but without the associated costs.

## Note for macOS Users

If you're using macOS, please use the macOS-specific version in the `macos` directory, which includes optimizations for the macOS environment and leverages native macOS tools for better extraction results.

## Future Improvements

- Add support for parallel processing to increase throughput
- Implement more sophisticated text cleaning for better quality
- Create a GUI interface for easier use
- Add support for extracting and preserving document structure
