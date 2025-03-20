# JFK Files PDF Extraction Solution

This solution provides a robust method for extracting text from the JFK Files PDF documents without requiring the Gemini API. It leverages multiple specialized PDF libraries to access the embedded OCR text that already exists in the PDFs.

## The Problem

The JFK Files collection consists of thousands of PDF documents that contain valuable historical information. While these PDFs already contain embedded OCR text, standard extraction methods (PyPDF2, pdfplumber, etc.) often fail to properly access this text. Previously, the Gemini API was used as the only reliable method to extract this text, but this approach is not sustainable for processing the entire collection due to cost constraints.

## The Solution

This repository provides a Python-based solution that:

1. Uses multiple specialized PDF libraries to extract embedded OCR text
2. Tries different extraction methods and selects the best result
3. Works on macOS without requiring cloud-based APIs
4. Processes files in batches to manage system resources
5. Provides detailed extraction metadata for analysis

## Key Components

- `extract_embedded_text.py`: Main script for extracting embedded OCR text from PDFs
- Test files and documentation

## Installation

### Prerequisites

On macOS:
```bash
# Install required system dependencies
brew install poppler tesseract

# Install Python dependencies
pip install pikepdf pymupdf pdfminer.six tqdm
```

On Ubuntu/Linux:
```bash
# Install required system dependencies
sudo apt-get install -y poppler-utils tesseract-ocr

# Install Python dependencies
pip install pikepdf pymupdf pdfminer.six tqdm
```

## Usage

```bash
# Basic usage
python extract_embedded_text.py --input /path/to/pdf/files --output /path/to/output/directory

# Process in smaller batches (recommended for large collections)
python extract_embedded_text.py --input /path/to/pdf/files --output /path/to/output/directory --batch 10
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
- **Scalability**: Can process the entire collection on a standard computer

## Performance

In testing, this solution successfully extracted text from PDFs that other methods failed to process. The extraction quality is comparable to the Gemini API approach but without the associated costs.

## Future Improvements

- Add support for parallel processing to increase throughput
- Implement more sophisticated text cleaning for better quality
- Create a GUI interface for easier use
- Add support for extracting and preserving document structure
