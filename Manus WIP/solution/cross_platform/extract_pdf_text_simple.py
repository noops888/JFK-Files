#!/usr/bin/env python3
"""
JFK Files PDF Extraction Tool - Simple Direct Text Extraction

This script extracts text from JFK Files PDFs using direct text extraction methods
that match how PDF viewers access the text layer.

Requirements:
- PyPDF2 or PyMuPDF (fitz)
- pdfminer.six (as fallback)

Usage:
python extract_pdf_text_simple.py [--input INPUT_DIR] [--output OUTPUT_DIR] [--debug]
"""

import os
import sys
import argparse
import time
import json
import platform
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Try to import required libraries
try:
    import fitz  # PyMuPDF
    HAVE_PYMUPDF = True
except ImportError:
    HAVE_PYMUPDF = False
    print("Warning: PyMuPDF not installed. Install with: pip install pymupdf")

try:
    import PyPDF2
    HAVE_PYPDF2 = True
except ImportError:
    HAVE_PYPDF2 = False
    print("Warning: PyPDF2 not installed. Install with: pip install PyPDF2")

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
    HAVE_PDFMINER = True
except ImportError:
    HAVE_PDFMINER = False
    print("Warning: pdfminer.six not installed. Install with: pip install pdfminer.six")

def extract_text_with_pymupdf(pdf_path, debug=False):
    """Extract text using PyMuPDF (fitz) - matches how PDF viewers access text"""
    if not HAVE_PYMUPDF:
        if debug:
            print("PyMuPDF not installed, skipping this method")
        return ""
    
    try:
        text = ""
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc):
                if debug:
                    print(f"Processing page {page_num+1}/{len(doc)} with PyMuPDF")
                
                # Extract text using the standard text extraction method
                # This matches how PDF viewers access the text layer
                page_text = page.get_text()
                text += page_text + "\n\n"
        
        if debug:
            print(f"Extracted {len(text)} characters with PyMuPDF")
        
        return text
    except Exception as e:
        if debug:
            print(f"Error extracting text with PyMuPDF: {str(e)}")
        return ""

def extract_text_with_pypdf2(pdf_path, debug=False):
    """Extract text using PyPDF2 - simple direct text extraction"""
    if not HAVE_PYPDF2:
        if debug:
            print("PyPDF2 not installed, skipping this method")
        return ""
    
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            
            for page_num in range(num_pages):
                if debug:
                    print(f"Processing page {page_num+1}/{num_pages} with PyPDF2")
                
                # Extract text from the page
                page = reader.pages[page_num]
                page_text = page.extract_text()
                text += page_text + "\n\n"
        
        if debug:
            print(f"Extracted {len(text)} characters with PyPDF2")
        
        return text
    except Exception as e:
        if debug:
            print(f"Error extracting text with PyPDF2: {str(e)}")
        return ""

def extract_text_with_pdfminer(pdf_path, debug=False):
    """Extract text using pdfminer.six - more advanced text extraction"""
    if not HAVE_PDFMINER:
        if debug:
            print("pdfminer.six not installed, skipping this method")
        return ""
    
    try:
        if debug:
            print(f"Extracting text with pdfminer.six")
        
        # Extract text using pdfminer.six
        text = pdfminer_extract_text(pdf_path)
        
        if debug:
            print(f"Extracted {len(text)} characters with pdfminer.six")
        
        return text
    except Exception as e:
        if debug:
            print(f"Error extracting text with pdfminer.six: {str(e)}")
        return ""

def process_pdf_file(pdf_path, output_dir, debug=False):
    """Process a PDF file to extract text"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Output file paths
        output_text_path = os.path.join(output_dir, f"{pdf_path.stem}.txt")
        output_json_path = os.path.join(output_dir, f"{pdf_path.stem}.json")
        
        # Skip if already processed
        if os.path.exists(output_text_path) and os.path.getsize(output_text_path) > 0:
            if debug:
                print(f"Skipping {pdf_path.name} - already processed")
            return True
        
        if debug:
            print(f"\nProcessing {pdf_path.name}...")
        else:
            print(f"Processing {pdf_path.name}...")
        
        # Dictionary to store extraction results
        extraction_results = {}
        
        # Try PyMuPDF first (usually gives the best results)
        if HAVE_PYMUPDF:
            if debug:
                print("Trying PyMuPDF...")
            pymupdf_text = extract_text_with_pymupdf(pdf_path, debug)
            extraction_results["pymupdf"] = len(pymupdf_text)
            
            if len(pymupdf_text) > 100:  # Consider it successful if we got some text
                # Save the extracted text
                with open(output_text_path, 'w', encoding='utf-8') as f:
                    f.write(pymupdf_text)
                
                # Save extraction metadata
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "filename": pdf_path.name,
                        "extraction_method": "pymupdf",
                        "characters": len(pymupdf_text),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }, f, indent=2)
                
                print(f"Successfully extracted {len(pymupdf_text)} characters from {pdf_path.name} using PyMuPDF")
                return True
        
        # Try PyPDF2 if PyMuPDF failed or is not available
        if HAVE_PYPDF2:
            if debug:
                print("Trying PyPDF2...")
            pypdf2_text = extract_text_with_pypdf2(pdf_path, debug)
            extraction_results["pypdf2"] = len(pypdf2_text)
            
            if len(pypdf2_text) > 100:  # Consider it successful if we got some text
                # Save the extracted text
                with open(output_text_path, 'w', encoding='utf-8') as f:
                    f.write(pypdf2_text)
                
                # Save extraction metadata
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "filename": pdf_path.name,
                        "extraction_method": "pypdf2",
                        "characters": len(pypdf2_text),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }, f, indent=2)
                
                print(f"Successfully extracted {len(pypdf2_text)} characters from {pdf_path.name} using PyPDF2")
                return True
        
        # Try pdfminer.six as a last resort
        if HAVE_PDFMINER:
            if debug:
                print("Trying pdfminer.six...")
            pdfminer_text = extract_text_with_pdfminer(pdf_path, debug)
            extraction_results["pdfminer"] = len(pdfminer_text)
            
            if len(pdfminer_text) > 100:  # Consider it successful if we got some text
                # Save the extracted text
                with open(output_text_path, 'w', encoding='utf-8') as f:
                    f.write(pdfminer_text)
                
                # Save extraction metadata
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "filename": pdf_path.name,
                        "extraction_method": "pdfminer",
                        "characters": len(pdfminer_text),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }, f, indent=2)
                
                print(f"Successfully extracted {len(pdfminer_text)} characters from {pdf_path.name} using pdfminer.six")
                return True
        
        # If all methods failed, find the best one and use it anyway
        best_method = max(extraction_results.items(), key=lambda x: x[1]) if extraction_results else (None, 0)
        method_name, char_count = best_method
        
        if method_name and char_count > 0:
            if method_name == "pymupdf":
                text = pymupdf_text
            elif method_name == "pypdf2":
                text = pypdf2_text
            elif method_name == "pdfminer":
                text = pdfminer_text
            else:
                text = ""
            
            # Save the extracted text
            with open(output_text_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            # Save extraction metadata
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "filename": pdf_path.name,
                    "extraction_method": method_name,
                    "characters": char_count,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }, f, indent=2)
            
            print(f"Extracted {char_count} characters from {pdf_path.name} using {method_name} (best available method)")
            return True
        else:
            # Save extraction metadata
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "filename": pdf_path.name,
                    "extraction_results": extraction_results,
                    "error": "No text extracted with any method",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }, f, indent=2)
            
            print(f"Error: Could not extract text from {pdf_path.name} using any method")
            return False
    
    except Exception as e:
        print(f"Error processing {pdf_path.name}: {str(e)}")
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract text from JFK Files PDFs using direct text extraction')
    parser.add_argument('--input', default='pdf_files', help='Input directory containing PDF files')
    parser.add_argument('--output', default='extracted_text', help='Output directory for extracted text')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
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
    
    # Print system information
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    
    # Check for required libraries
    if not (HAVE_PYMUPDF or HAVE_PYPDF2 or HAVE_PDFMINER):
        print("Error: No PDF extraction libraries available. Please install at least one of: pymupdf, PyPDF2, pdfminer.six")
        return
    
    # Process files
    successful = 0
    failed = 0
    
    # Process files in parallel
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Submit tasks
        future_to_pdf = {
            executor.submit(
                process_pdf_file, 
                pdf_file, 
                output_dir, 
                args.debug
            ): pdf_file for pdf_file in pdf_files
        }
        
        # Process results as they complete
        for future in as_completed(future_to_pdf):
            pdf_file = future_to_pdf[future]
            try:
                result = future.result()
                if result:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"Error processing {pdf_file.name}: {str(e)}")
                failed += 1
    
    # Print final summary
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful} files")
    print(f"Failed: {failed} files")
    print(f"Success rate: {successful/total_files*100:.2f}%")

if __name__ == "__main__":
    main()
