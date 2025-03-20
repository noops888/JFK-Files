# JFK Files Processing Project

This repository contains a comprehensive solution for processing the JFK Files collection (65.9 GB with 73,056 files) and making it queryable through an LLM system.

## Background

The JFK Files collection consists of:
- 2025: 2,182 files (6.47 GB)
- 2023: 2,682 files (6.35 GB)
- 2022: 13,201 files (14.15 GB)
- 2021: 1,487 files (1.36 GB)
- 2017-2018: 53,499 files (37.76 GB)

Most of these files are PDFs that have already been OCR'd, but extracting the text programmatically has proven challenging with standard libraries.

## Repository Contents

- `jfk_files_fast_extraction_pipeline.md`: Streamlined approach to quickly extract text from files that already have OCR
- `jfk_files_secondary_processing_pipeline.md`: Thorough approach for problematic documents
- `jfk_files_dual_approach_implementation_guide.md`: Comprehensive step-by-step instructions with code examples
- `jfk_files_updated_recommendations.md`: Key recommendations, limitations, and alternatives
- `jfk_files_dual_approach_summary.md`: Concise overview of the complete strategy

## Extraction Approaches

### Fast Path Extraction

Initial attempts using standard PDF libraries (PyPDF2, pdfplumber, PyMuPDF) were unsuccessful in extracting the embedded OCR text from the PDFs.

### Gemini API Approach

After examining successful approaches, we found that using Google's Gemini API is effective for extracting text from these PDFs. The script below demonstrates this approach:

```python
import os
import time
from google import genai
from google.genai import types
from pathlib import Path

def convert_pdf_to_text(pdf_path, retries=3, initial_delay=1):
    try:
        # Extract filename without extension
        filename = Path(pdf_path).stem
        output_path = Path("extracted_text") / f"{filename}.txt"
        
        # Skip if already converted
        if output_path.exists():
            print(f"Skipping {filename} - already converted")
            return
        
        # Initialize Gemini API client
        client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY"),
        )
        
        for attempt in range(retries):
            try:
                # Upload the PDF file
                uploaded_file = client.files.upload(file=str(pdf_path))
                
                # Use Gemini Flash model
                model = "gemini-2.0-flash"
                
                # Create the prompt
                contents = [
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_uri(
                                file_uri=uploaded_file.uri,
                                mime_type=uploaded_file.mime_type,
                            ),
                            types.Part.from_text(text="Extract all text from this PDF document. Return only the raw text without any formatting or commentary."),
                        ],
                    ),
                ]
                
                # Configure generation parameters
                generate_content_config = types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="text/plain",
                )
                
                # Create output directory if it doesn't exist
                os.makedirs(output_path.parent, exist_ok=True)
                
                # Process and collect the response
                full_response = ""
                for chunk in client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=generate_content_config,
                ):
                    full_response += chunk.text
                
                # Only write if response is not empty
                if full_response.strip():
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(full_response)
                    print(f"Converted {filename}")
                else:
                    print(f"Skipping {filename} - empty response")
                    raise Exception("Empty response from Gemini")
                
                return
                
            except Exception as e:
                if attempt < retries - 1:
                    delay = initial_delay * (2 ** attempt)  # Exponential backoff
                    time.sleep(delay)
                    print(f"Retrying {filename} after error: {e}")
                else:
                    print(f"Failed to process {filename} after {retries} attempts: {e}")
                    raise
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")

def main():
    # Get list of PDF files
    pdf_dir = Path("original_files")
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    
    # Process files one by one (can be parallelized with ThreadPoolExecutor)
    for pdf_file in pdf_files:
        convert_pdf_to_text(pdf_file)

if __name__ == "__main__":
    main()
```

### OCR Approach (Alternative)

If the Gemini API approach is not feasible, a direct OCR approach can be used:

```python
import os
import subprocess
from pathlib import Path
import re
from tqdm import tqdm

# Configuration
ORIGINAL_FILES_DIR = "original_files"
EXTRACTED_TEXT_DIR = "extracted_text"
DATABASE_DIR = "database"

def extract_text_with_ocr(pdf_path):
    """Extract text using OCR via Tesseract"""
    try:
        # Create a temporary file for the output
        output_file = "temp_ocr.txt"
        
        # Run OCR directly on the PDF using the pdf2ppm and tesseract
        cmd = f"pdfimages -j '{pdf_path}' temp_img && for f in temp_img-*; do tesseract $f temp_ocr_part -l eng; cat temp_ocr_part.txt >> {output_file}; done"
        subprocess.run(cmd, shell=True, check=True)
        
        # Read the extracted text
        with open(output_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Clean up temporary files
        subprocess.run("rm -f temp_img-* temp_ocr_part.txt temp_ocr.txt", shell=True)
        
        return text
    except Exception as e:
        print(f"OCR error with {pdf_path}: {str(e)}")
        return ""
```

## Next Steps After Extraction

Once text extraction is complete, follow these steps:

1. **Chunking**: Split documents into appropriate chunks (500-1000 tokens)
2. **Embedding**: Generate embeddings for each chunk using a model like Sentence-Transformers
3. **Vector Storage**: Store chunks and embeddings in a vector database like ChromaDB
4. **Query Interface**: Create a simple interface to query the knowledge base

## Implementation Timeline

| Stage | Estimated Time | Technical Complexity |
|-------|----------------|----------------------|
| Text Extraction | 1-3 days | Low to Medium |
| Chunking & Embedding | 2-3 days | Medium |
| Query Interface | 1 day | Low |
| **Total** | **4-7 days** | **Low to Medium** |

## References

- Original GitHub repository for JFK Files downloading: https://github.com/noops888/JFK-Files
- Successful text extraction approach: https://github.com/amasad/jfk_files
