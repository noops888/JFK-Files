#!/usr/bin/env python3
"""
JFK Files PDF Extraction Tool using OCR

This script extracts text from PDF files using an OCR-based approach that works
without requiring the Gemini API. It converts PDF pages to images and then applies
Tesseract OCR to extract the text.

Requirements:
- pdf2image (requires poppler-utils on Linux/Mac)
- pytesseract (requires tesseract-ocr to be installed)
- Pillow
- tqdm

Installation on macOS:
brew install poppler tesseract
pip install pdf2image pytesseract pillow tqdm

Usage:
python extract_with_ocr.py [--input INPUT_DIR] [--output OUTPUT_DIR] [--dpi DPI] [--batch BATCH_SIZE]
"""

import os
import argparse
import time
import re
import concurrent.futures
from pathlib import Path
from tqdm import tqdm

# Import required libraries with error handling
try:
    import pdf2image
except ImportError:
    print("Error: pdf2image library not found. Please install it using:")
    print("pip install pdf2image")
    print("Also ensure poppler-utils is installed:")
    print("  - On macOS: brew install poppler")
    print("  - On Ubuntu: sudo apt-get install poppler-utils")
    exit(1)

try:
    import pytesseract
    from PIL import Image
except ImportError:
    print("Error: pytesseract or Pillow library not found. Please install them using:")
    print("pip install pytesseract pillow")
    print("Also ensure tesseract-ocr is installed:")
    print("  - On macOS: brew install tesseract")
    print("  - On Ubuntu: sudo apt-get install tesseract-ocr")
    exit(1)

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

def extract_text_from_pdf(pdf_path, output_path, dpi=300):
    """Extract text from a PDF file using OCR"""
    try:
        print(f"Processing {pdf_path.name}...")
        
        # Skip if already processed and not empty
        if output_path.exists() and os.path.getsize(output_path) > 100:
            print(f"Skipping {pdf_path.name} - already processed")
            return True
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path.parent, exist_ok=True)
        
        # Convert PDF to images
        print(f"Converting {pdf_path.name} to images (DPI={dpi})...")
        try:
            images = pdf2image.convert_from_path(pdf_path, dpi=dpi)
        except Exception as e:
            print(f"Error converting PDF to images: {str(e)}")
            # Try with lower DPI if conversion fails
            try:
                print(f"Retrying with lower DPI (150)...")
                images = pdf2image.convert_from_path(pdf_path, dpi=150)
            except Exception as e:
                print(f"Error converting PDF to images even with lower DPI: {str(e)}")
                return False
        
        print(f"Converted {len(images)} pages to images")
        
        # Extract text using OCR
        full_text = ""
        for i, img in enumerate(images):
            print(f"Applying OCR to page {i+1}/{len(images)}...")
            try:
                page_text = pytesseract.image_to_string(img)
                full_text += page_text + "\n\n"
            except Exception as e:
                print(f"Error during OCR on page {i+1}: {str(e)}")
                # Continue with other pages even if one fails
        
        # Clean the text
        cleaned_text = clean_text(full_text)
        
        # Save the extracted text
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        
        print(f"Successfully extracted {len(cleaned_text)} characters from {pdf_path.name}")
        return True
    
    except Exception as e:
        print(f"Error processing {pdf_path.name}: {str(e)}")
        return False

def process_batch(batch_files, output_dir, dpi):
    """Process a batch of PDF files"""
    results = []
    for pdf_file in batch_files:
        output_path = Path(output_dir) / f"{pdf_file.stem}.txt"
        result = extract_text_from_pdf(pdf_file, output_path, dpi)
        results.append((pdf_file, result))
    return results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract text from PDF files using OCR')
    parser.add_argument('--input', default='original_files', help='Input directory containing PDF files')
    parser.add_argument('--output', default='extracted_text', help='Output directory for extracted text')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for PDF to image conversion (higher is better quality but slower)')
    parser.add_argument('--batch', type=int, default=10, help='Number of files to process in each batch')
    args = parser.parse_args()
    
    # Get list of PDF files
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all PDF files
    pdf_files = list(input_dir.glob("**/*.pdf"))
    total_files = len(pdf_files)
    
    if total_files == 0:
        print(f"No PDF files found in {input_dir}")
        return
    
    print(f"Found {total_files} PDF files to process")
    
    # Process files in batches
    successful = 0
    failed = 0
    
    # Process files in batches
    batch_size = min(args.batch, total_files)
    batches = [pdf_files[i:i + batch_size] for i in range(0, total_files, batch_size)]
    
    for i, batch in enumerate(batches):
        print(f"\nProcessing batch {i+1}/{len(batches)} ({len(batch)} files)...")
        
        # Process batch
        for pdf_file in tqdm(batch, desc="Processing PDFs"):
            output_path = output_dir / f"{pdf_file.stem}.txt"
            if extract_text_from_pdf(pdf_file, output_path, args.dpi):
                successful += 1
            else:
                failed += 1
        
        # Print batch summary
        print(f"Batch {i+1} complete: {successful} successful, {failed} failed")
        
        # Add a pause between batches to prevent system overload
        if i < len(batches) - 1:
            print(f"Pausing for 5 seconds before next batch...")
            time.sleep(5)
    
    # Print final summary
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful} files")
    print(f"Failed: {failed} files")
    print(f"Success rate: {successful/total_files*100:.2f}%")

if __name__ == "__main__":
    main()
