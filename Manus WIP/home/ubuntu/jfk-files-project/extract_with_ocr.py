import os
import subprocess
from pathlib import Path
import re
from tqdm import tqdm

# Configuration
ORIGINAL_FILES_DIR = "original_files"
EXTRACTED_TEXT_DIR = "extracted_text"

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

def clean_text(text):
    """Basic cleaning of extracted text"""
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common OCR errors
    text = re.sub(r'([a-z])\s+([a-z])', r'\1\2', text)  # Fix split words
    
    # Remove non-printable characters
    text = re.sub(r'[^\x20-\x7E\n]', '', text)
    
    # Normalize line breaks
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def process_pdf_file(pdf_path):
    """Process a single PDF file"""
    # Create output filename
    rel_path = pdf_path.relative_to(Path(ORIGINAL_FILES_DIR))
    output_path = Path(EXTRACTED_TEXT_DIR) / rel_path.with_suffix('.txt')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Extract text with OCR
    text = extract_text_with_ocr(pdf_path)
    
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Save whatever text we got
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
    
    # Return success if we got any text at all
    if len(cleaned_text.strip()) > 0:
        return True
    else:
        print(f"Warning: No text extracted from {pdf_path}")
        return False

def main():
    # Get list of PDF files
    original_dir = Path(ORIGINAL_FILES_DIR)
    
    # Create directories if they don't exist
    os.makedirs(EXTRACTED_TEXT_DIR, exist_ok=True)
    
    # Find all PDF files
    pdf_files = list(original_dir.glob("**/*.pdf"))
    
    # Limit to first 5 files for testing
    # Remove this line to process all files
    pdf_files = pdf_files[:5]
    
    total_files = len(pdf_files)
    print(f"Found {total_files} PDF files to process")
    
    # Process files with progress bar
    successful = 0
    problematic = 0
    
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        result = process_pdf_file(pdf_file)
        if result:
            successful += 1
        else:
            problematic += 1
    
    print(f"Processing complete!")
    print(f"Successfully processed: {successful} files")
    print(f"Problematic files: {problematic} files")
    print(f"Success rate: {successful/total_files*100:.2f}%")

if __name__ == "__main__":
    main()
