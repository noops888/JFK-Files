import os
import PyPDF2
import pdfplumber
from pathlib import Path
import config
import re
from tqdm import tqdm

def extract_text_with_pypdf2(pdf_path):
    """Extract text using PyPDF2"""
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n\n"
        return text
    except Exception as e:
        print(f"PyPDF2 error with {pdf_path}: {str(e)}")
        return ""

def extract_text_with_pdfplumber(pdf_path):
    """Extract text using pdfplumber"""
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n\n"
        return text
    except Exception as e:
        print(f"pdfplumber error with {pdf_path}: {str(e)}")
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

def process_pdf_file(pdf_path, output_dir):
    """Process a single PDF file"""
    # Create output filename
    rel_path = pdf_path.relative_to(Path(config.ORIGINAL_FILES_DIR))
    output_path = Path(output_dir) / rel_path.with_suffix('.txt')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Extract text with PyPDF2 first
    text = extract_text_with_pypdf2(pdf_path)
    
    # If not enough text, try pdfplumber
    if len(text.strip()) < config.MIN_TEXT_LENGTH:
        text = extract_text_with_pdfplumber(pdf_path)
    
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Check if we got enough text
    if len(cleaned_text.strip()) < config.MIN_TEXT_LENGTH:
        print(f"Warning: Not enough text extracted from {pdf_path}")
        return False
    
    # Write the text to the output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
    
    return True

def main():
    # Get all PDF files in the original directory
    original_dir = Path(config.ORIGINAL_FILES_DIR)
    extracted_dir = Path(config.EXTRACTED_TEXT_DIR)
    
    # Create directories if they don't exist
    os.makedirs(extracted_dir, exist_ok=True)
    
    # Find all PDF files
    pdf_files = list(original_dir.glob("**/*.pdf"))
    total_files = len(pdf_files)
    print(f"Found {total_files} PDF files to process")
    
    # Process files with progress bar
    successful = 0
    problematic = 0
    
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        result = process_pdf_file(pdf_file, extracted_dir)
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
