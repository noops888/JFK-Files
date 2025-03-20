import os
import fitz  # PyMuPDF
import subprocess
from pathlib import Path
from tqdm import tqdm
import concurrent.futures

# Configuration
ORIGINAL_FILES_DIR = "original_files"
EXTRACTED_TEXT_DIR = "extracted_text"
MAX_WORKERS = 4  # Reduced for better debugging

# Create output directory
os.makedirs(EXTRACTED_TEXT_DIR, exist_ok=True)

def extract_text_with_pymupdf(pdf_path):
    """Extract text using PyMuPDF (primary method)"""
    try:
        text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                page_text = page.get_text()
                text += page_text + "\n\n"
        return text
    except Exception as e:
        print(f"PyMuPDF error with {pdf_path}: {str(e)}")
        return ""

def extract_text_with_pdftotext(pdf_path):
    """Extract text using pdftotext (fallback method)"""
    try:
        result = subprocess.run(
            ['pdftotext', '-layout', str(pdf_path), '-'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except Exception as e:
        print(f"pdftotext error with {pdf_path}: {str(e)}")
        return ""

def extract_text_with_pdftotext_raw(pdf_path):
    """Extract text using pdftotext with raw option (second fallback)"""
    try:
        result = subprocess.run(
            ['pdftotext', '-raw', str(pdf_path), '-'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except Exception as e:
        print(f"pdftotext raw error with {pdf_path}: {str(e)}")
        return ""

def process_pdf_file(pdf_path):
    """Process a single PDF file with multiple methods and debugging"""
    try:
        # Get filename
        filename = pdf_path.stem
        output_path = Path(EXTRACTED_TEXT_DIR) / f"{filename}.txt"
        
        # Skip if already processed and not empty
        if output_path.exists() and os.path.getsize(output_path) > 0:
            print(f"Skipping {filename} - already processed")
            return True
            
        print(f"Processing {filename}...")
        
        # Try PyMuPDF first
        text = extract_text_with_pymupdf(pdf_path)
        method = "PyMuPDF"
        
        # If not enough text, try pdftotext with layout
        if len(text.strip()) < 100:
            print(f"  PyMuPDF extracted only {len(text.strip())} chars, trying pdftotext...")
            text = extract_text_with_pdftotext(pdf_path)
            method = "pdftotext-layout"
        
        # If still not enough text, try pdftotext with raw option
        if len(text.strip()) < 100:
            print(f"  pdftotext-layout extracted only {len(text.strip())} chars, trying pdftotext-raw...")
            text = extract_text_with_pdftotext_raw(pdf_path)
            method = "pdftotext-raw"
        
        # Save the extracted text
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        # Check if we got any text
        if len(text.strip()) > 0:
            print(f"  Successfully extracted {len(text.strip())} chars with {method} from {filename}")
            return True
        else:
            print(f"  WARNING: All extraction methods failed for {filename}")
            return False
        
    except Exception as e:
        print(f"Error processing {pdf_path.name}: {str(e)}")
        return False

def main():
    # Get list of PDF files
    pdf_dir = Path(ORIGINAL_FILES_DIR)
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    
    # Limit to 10 files for testing
    pdf_files = pdf_files[:10]
    
    print(f"Found {len(pdf_files)} PDF files")
    
    # Process files sequentially for better debugging
    successful = 0
    failed = 0
    
    for pdf_file in tqdm(pdf_files):
        if process_pdf_file(pdf_file):
            successful += 1
        else:
            failed += 1
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful} files")
    print(f"Failed: {failed} files")
    print(f"Success rate: {successful/len(pdf_files)*100:.2f}%")

if __name__ == "__main__":
    main()
