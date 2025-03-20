#!/usr/bin/env python3
"""
JFK Files PDF Extraction Tool for Embedded OCR Text

This script extracts embedded OCR text from PDF files without requiring the Gemini API.
It uses multiple specialized PDF libraries to access the text layer that already exists
in the PDFs rather than performing unnecessary OCR.

Requirements:
- pikepdf
- pdfminer.six
- pymupdf (PyMuPDF)
- tqdm

Installation on macOS:
brew install poppler
pip install pikepdf pdfminer.six pymupdf tqdm

Usage:
python extract_embedded_text.py [--input INPUT_DIR] [--output OUTPUT_DIR] [--batch BATCH_SIZE]
"""

import os
import argparse
import time
import re
import json
from pathlib import Path
from tqdm import tqdm

# Import required libraries with error handling
try:
    import pikepdf
except ImportError:
    print("Error: pikepdf library not found. Please install it using:")
    print("pip install pikepdf")
    exit(1)

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Error: PyMuPDF library not found. Please install it using:")
    print("pip install pymupdf")
    exit(1)

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
    from pdfminer.pdfdocument import PDFDocument
    from pdfminer.pdfparser import PDFParser
except ImportError:
    print("Error: pdfminer.six library not found. Please install it using:")
    print("pip install pdfminer.six")
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

def extract_text_with_pikepdf(pdf_path):
    """Extract text using pikepdf to access PDF structure"""
    try:
        text = ""
        with pikepdf.open(pdf_path) as pdf:
            # Extract text from each page
            for page_num, page in enumerate(pdf.pages):
                if '/Contents' in page:
                    # Try to extract text from content stream
                    content = page['/Contents']
                    if isinstance(content, pikepdf.Array):
                        for item in content:
                            stream_text = item.read_bytes().decode('utf-8', errors='ignore')
                            text += stream_text + "\n"
                    else:
                        stream_text = content.read_bytes().decode('utf-8', errors='ignore')
                        text += stream_text + "\n"
                
                # Try to extract text from annotations
                if '/Annots' in page:
                    annots = page['/Annots']
                    if isinstance(annots, pikepdf.Array):
                        for annot in annots:
                            if '/Contents' in annot:
                                annot_text = str(annot['/Contents'])
                                text += annot_text + "\n"
        
        return text
    except Exception as e:
        print(f"pikepdf error with {pdf_path}: {str(e)}")
        return ""

def extract_text_with_pymupdf(pdf_path):
    """Extract text using PyMuPDF (fitz)"""
    try:
        text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                # Try different text extraction methods
                page_text = page.get_text("text")  # Standard text extraction
                if len(page_text.strip()) < 100:
                    # Try alternative extraction methods
                    page_text = page.get_text("blocks")  # Extract text blocks
                    if isinstance(page_text, list):
                        page_text = "\n".join([block[4] for block in page_text if len(block) > 4])
                
                text += page_text + "\n\n"
        return text
    except Exception as e:
        print(f"PyMuPDF error with {pdf_path}: {str(e)}")
        return ""

def extract_text_with_pdfminer(pdf_path):
    """Extract text using pdfminer.six"""
    try:
        # Use high-level function for text extraction
        text = pdfminer_extract_text(pdf_path)
        
        # If high-level extraction fails, try low-level approach
        if len(text.strip()) < 100:
            with open(pdf_path, 'rb') as file:
                parser = PDFParser(file)
                doc = PDFDocument(parser)
                # Check if the document has text extraction permission
                if 'ExtractText' in doc.get_outlines():
                    print(f"Note: {pdf_path} has text extraction restrictions")
        
        return text
    except Exception as e:
        print(f"pdfminer error with {pdf_path}: {str(e)}")
        return ""

def extract_text_with_poppler(pdf_path):
    """Extract text using poppler-utils (pdftotext)"""
    try:
        import subprocess
        # Try with layout preservation
        result = subprocess.run(
            ['pdftotext', '-layout', str(pdf_path), '-'],
            capture_output=True,
            text=True,
            check=False  # Don't raise exception on non-zero exit
        )
        
        text = result.stdout
        
        # If layout mode doesn't yield good results, try raw mode
        if len(text.strip()) < 100:
            result = subprocess.run(
                ['pdftotext', '-raw', str(pdf_path), '-'],
                capture_output=True,
                text=True,
                check=False
            )
            text = result.stdout
            
        return text
    except Exception as e:
        print(f"poppler error with {pdf_path}: {str(e)}")
        return ""

def extract_embedded_text(pdf_path, output_path):
    """Extract embedded text from a PDF file using multiple methods"""
    try:
        print(f"Processing {pdf_path.name}...")
        
        # Skip if already processed and not empty
        if output_path.exists() and os.path.getsize(output_path) > 100:
            print(f"Skipping {pdf_path.name} - already processed")
            return True
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path.parent, exist_ok=True)
        
        # Try multiple extraction methods and use the best result
        extraction_results = {}
        
        # Method 1: PyMuPDF (usually fastest and most reliable)
        print(f"Extracting text from {pdf_path.name} using PyMuPDF...")
        pymupdf_text = extract_text_with_pymupdf(pdf_path)
        extraction_results["pymupdf"] = len(pymupdf_text.strip())
        
        # Method 2: pdfminer.six (good for complex PDFs)
        print(f"Extracting text from {pdf_path.name} using pdfminer.six...")
        pdfminer_text = extract_text_with_pdfminer(pdf_path)
        extraction_results["pdfminer"] = len(pdfminer_text.strip())
        
        # Method 3: poppler-utils (pdftotext, good for simple PDFs)
        print(f"Extracting text from {pdf_path.name} using poppler-utils...")
        poppler_text = extract_text_with_poppler(pdf_path)
        extraction_results["poppler"] = len(poppler_text.strip())
        
        # Method 4: pikepdf (good for accessing PDF internals)
        print(f"Extracting text from {pdf_path.name} using pikepdf...")
        pikepdf_text = extract_text_with_pikepdf(pdf_path)
        extraction_results["pikepdf"] = len(pikepdf_text.strip())
        
        # Determine which method extracted the most text
        best_method = max(extraction_results, key=extraction_results.get)
        print(f"Best extraction method for {pdf_path.name}: {best_method} ({extraction_results[best_method]} chars)")
        
        # Use the text from the best method
        if best_method == "pymupdf":
            text = pymupdf_text
        elif best_method == "pdfminer":
            text = pdfminer_text
        elif best_method == "poppler":
            text = poppler_text
        else:  # pikepdf
            text = pikepdf_text
        
        # Clean the text
        cleaned_text = clean_text(text)
        
        # Save the extracted text
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        
        # Save extraction metadata
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                "filename": pdf_path.name,
                "extraction_results": extraction_results,
                "best_method": best_method,
                "extracted_chars": len(cleaned_text),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=2)
        
        print(f"Successfully extracted {len(cleaned_text)} characters from {pdf_path.name}")
        return True
    
    except Exception as e:
        print(f"Error processing {pdf_path.name}: {str(e)}")
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract embedded OCR text from PDF files')
    parser.add_argument('--input', default='original_files', help='Input directory containing PDF files')
    parser.add_argument('--output', default='extracted_text', help='Output directory for extracted text')
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
            if extract_embedded_text(pdf_file, output_path):
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
