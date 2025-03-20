#!/usr/bin/env python3
"""
JFK Files PDF Extraction Tool for Embedded OCR Text - Enhanced for macOS

This script extracts embedded OCR text from PDF files without requiring the Gemini API.
It uses multiple specialized PDF libraries and techniques optimized for macOS to access
the text layer that already exists in the PDFs.

Requirements:
- pikepdf
- pdfminer.six
- pymupdf (PyMuPDF)
- pdfplumber
- tqdm

Installation on macOS:
brew install poppler tesseract
pip install pikepdf pymupdf pdfminer.six pdfplumber tqdm

Usage:
python extract_embedded_text_macos.py [--input INPUT_DIR] [--output OUTPUT_DIR] [--batch BATCH_SIZE] [--debug]
"""

import os
import argparse
import time
import re
import json
import subprocess
import tempfile
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

try:
    import pdfplumber
except ImportError:
    print("Error: pdfplumber library not found. Please install it using:")
    print("pip install pdfplumber")
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

def extract_text_with_pymupdf(pdf_path, debug=False):
    """Extract text using PyMuPDF (fitz) with enhanced options for macOS"""
    try:
        text = ""
        with fitz.open(pdf_path) as doc:
            # Try different extraction methods
            for page_num, page in enumerate(doc):
                if debug:
                    print(f"  PyMuPDF: Processing page {page_num+1}/{len(doc)}")
                
                # Try different text extraction methods
                methods = [
                    ("text", page.get_text("text")),
                    ("html", page.get_text("html")),
                    ("dict", page.get_text("dict")),
                    ("json", page.get_text("json")),
                    ("rawdict", page.get_text("rawdict")),
                    ("xhtml", page.get_text("xhtml"))
                ]
                
                # Find the method that extracts the most text
                best_method = max(methods, key=lambda x: len(x[1]))
                method_name, page_text = best_method
                
                if debug:
                    print(f"  PyMuPDF: Best method for page {page_num+1}: {method_name} ({len(page_text)} chars)")
                
                # If we got HTML or XML, extract just the text content
                if method_name in ["html", "xhtml"]:
                    # Simple regex to remove HTML tags
                    page_text = re.sub(r'<[^>]+>', ' ', page_text)
                
                # For dict and rawdict formats, extract just the text
                if method_name in ["dict", "rawdict", "json"]:
                    if isinstance(page_text, str):
                        # It's already a string (json format)
                        pass
                    elif isinstance(page_text, dict) and "blocks" in page_text:
                        # Handle dict format
                        blocks_text = []
                        for block in page_text.get("blocks", []):
                            if "lines" in block:
                                for line in block["lines"]:
                                    if "spans" in line:
                                        for span in line["spans"]:
                                            if "text" in span:
                                                blocks_text.append(span["text"])
                        page_text = " ".join(blocks_text)
                
                text += page_text + "\n\n"
        
        return text
    except Exception as e:
        if debug:
            print(f"PyMuPDF error with {pdf_path}: {str(e)}")
        return ""

def extract_text_with_pdfminer(pdf_path, debug=False):
    """Extract text using pdfminer.six with enhanced options for macOS"""
    try:
        # Use high-level function for text extraction with various parameters
        text = ""
        
        # Try different layout analysis parameters
        laparams_options = [
            {"line_margin": 0.5, "char_margin": 2.0, "word_margin": 0.1},
            {"line_margin": 0.3, "char_margin": 1.0, "word_margin": 0.2},
            {"line_margin": 0.2, "char_margin": 0.5, "word_margin": 0.1},
            None  # Default parameters
        ]
        
        best_text = ""
        best_params = None
        
        for params in laparams_options:
            try:
                from pdfminer.layout import LAParams
                laparams = LAParams(**params) if params else None
                
                # Extract with current parameters
                current_text = pdfminer_extract_text(pdf_path, laparams=laparams)
                
                if len(current_text) > len(best_text):
                    best_text = current_text
                    best_params = params
            except Exception as e:
                if debug:
                    print(f"  PDFMiner error with params {params}: {str(e)}")
        
        if debug and best_params:
            print(f"  PDFMiner: Best parameters: {best_params} ({len(best_text)} chars)")
        
        text = best_text
        
        # If high-level extraction fails, try low-level approach
        if len(text.strip()) < 100:
            if debug:
                print("  PDFMiner: High-level extraction yielded minimal text, trying low-level approach")
            
            with open(pdf_path, 'rb') as file:
                parser = PDFParser(file)
                doc = PDFDocument(parser)
                # Check if the document has text extraction permission
                if 'ExtractText' in doc.get_outlines():
                    if debug:
                        print(f"  PDFMiner: Note: {pdf_path} has text extraction restrictions")
        
        return text
    except Exception as e:
        if debug:
            print(f"PDFMiner error with {pdf_path}: {str(e)}")
        return ""

def extract_text_with_pdfplumber(pdf_path, debug=False):
    """Extract text using pdfplumber (works well on macOS)"""
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                if debug:
                    print(f"  PDFPlumber: Processing page {i+1}/{len(pdf.pages)}")
                
                # Try different extraction methods
                methods = []
                
                # Standard extraction
                try:
                    standard_text = page.extract_text() or ""
                    methods.append(("standard", standard_text))
                except Exception as e:
                    if debug:
                        print(f"  PDFPlumber standard extraction error: {str(e)}")
                
                # Table extraction
                try:
                    tables = page.extract_tables()
                    table_text = "\n".join(["\n".join([" | ".join([str(cell or "") for cell in row]) for row in table]) for table in tables])
                    methods.append(("tables", table_text))
                except Exception as e:
                    if debug:
                        print(f"  PDFPlumber table extraction error: {str(e)}")
                
                # Words extraction
                try:
                    words = page.extract_words()
                    words_text = " ".join([word.get("text", "") for word in words])
                    methods.append(("words", words_text))
                except Exception as e:
                    if debug:
                        print(f"  PDFPlumber words extraction error: {str(e)}")
                
                # Find the method that extracts the most text
                if methods:
                    best_method = max(methods, key=lambda x: len(x[1]))
                    method_name, page_text = best_method
                    
                    if debug:
                        print(f"  PDFPlumber: Best method for page {i+1}: {method_name} ({len(page_text)} chars)")
                    
                    text += page_text + "\n\n"
        
        return text
    except Exception as e:
        if debug:
            print(f"PDFPlumber error with {pdf_path}: {str(e)}")
        return ""

def extract_text_with_poppler(pdf_path, debug=False):
    """Extract text using poppler-utils (pdftotext) with enhanced options for macOS"""
    try:
        # Try different pdftotext options
        options = [
            ["-layout"],  # Maintain original layout
            ["-raw"],     # Raw text extraction
            ["-htmlmeta"],  # HTML with metadata
            ["-enc", "UTF-8"],  # Force UTF-8 encoding
            ["-eol", "unix"],  # Unix line endings
            ["-nopgbrk"],  # No page breaks
            ["-layout", "-enc", "UTF-8"],  # Combined options
            ["-raw", "-enc", "UTF-8"]  # Combined options
        ]
        
        best_text = ""
        best_option = None
        
        for option in options:
            try:
                cmd = ["pdftotext"] + option + [str(pdf_path), "-"]
                if debug:
                    print(f"  Poppler: Trying command: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False  # Don't raise exception on non-zero exit
                )
                
                if result.returncode == 0 and len(result.stdout) > len(best_text):
                    best_text = result.stdout
                    best_option = option
            except Exception as e:
                if debug:
                    print(f"  Poppler error with option {option}: {str(e)}")
        
        if debug and best_option:
            print(f"  Poppler: Best option: {best_option} ({len(best_text)} chars)")
        
        return best_text
    except Exception as e:
        if debug:
            print(f"Poppler error with {pdf_path}: {str(e)}")
        return ""

def extract_text_with_pikepdf(pdf_path, debug=False):
    """Extract text using pikepdf with enhanced options for macOS"""
    try:
        text = ""
        with pikepdf.open(pdf_path) as pdf:
            # Extract text from each page
            for page_num, page in enumerate(pdf.pages):
                if debug:
                    print(f"  Pikepdf: Processing page {page_num+1}/{len(pdf.pages)}")
                
                page_text = ""
                
                # Try to extract text from content stream
                if '/Contents' in page:
                    content = page['/Contents']
                    if isinstance(content, pikepdf.Array):
                        for item in content:
                            try:
                                stream_text = item.read_bytes().decode('utf-8', errors='ignore')
                                page_text += stream_text + "\n"
                            except Exception as e:
                                if debug:
                                    print(f"  Pikepdf content stream error: {str(e)}")
                    else:
                        try:
                            stream_text = content.read_bytes().decode('utf-8', errors='ignore')
                            page_text += stream_text + "\n"
                        except Exception as e:
                            if debug:
                                print(f"  Pikepdf content stream error: {str(e)}")
                
                # Try to extract text from annotations
                if '/Annots' in page:
                    annots = page['/Annots']
                    if isinstance(annots, pikepdf.Array):
                        for annot in annots:
                            try:
                                if '/Contents' in annot:
                                    annot_text = str(annot['/Contents'])
                                    page_text += annot_text + "\n"
                            except Exception as e:
                                if debug:
                                    print(f"  Pikepdf annotation error: {str(e)}")
                
                # Try to extract text from form fields
                if '/AcroForm' in pdf.Root:
                    try:
                        acroform = pdf.Root.AcroForm
                        if '/Fields' in acroform:
                            fields = acroform.Fields
                            if isinstance(fields, pikepdf.Array):
                                for field in fields:
                                    if '/V' in field:
                                        field_text = str(field.V)
                                        page_text += field_text + "\n"
                    except Exception as e:
                        if debug:
                            print(f"  Pikepdf form field error: {str(e)}")
                
                text += page_text + "\n\n"
        
        return text
    except Exception as e:
        if debug:
            print(f"Pikepdf error with {pdf_path}: {str(e)}")
        return ""

def extract_text_with_qpdf(pdf_path, debug=False):
    """Extract text using qpdf (good for encrypted or damaged PDFs)"""
    try:
        # Create a temporary file for the fixed PDF
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Try to repair/normalize the PDF with qpdf
        try:
            cmd = ["qpdf", "--replace-input", str(pdf_path), temp_path]
            if debug:
                print(f"  QPDF: Running command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode != 0 and debug:
                print(f"  QPDF warning: {result.stderr}")
        except Exception as e:
            if debug:
                print(f"  QPDF error: {str(e)}")
            os.unlink(temp_path)
            return ""
        
        # Now try to extract text from the repaired PDF using pdftotext
        try:
            cmd = ["pdftotext", "-layout", temp_path, "-"]
            if debug:
                print(f"  QPDF: Extracting text with command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            text = result.stdout
            os.unlink(temp_path)
            return text
        except Exception as e:
            if debug:
                print(f"  QPDF extraction error: {str(e)}")
            os.unlink(temp_path)
            return ""
    except Exception as e:
        if debug:
            print(f"QPDF error with {pdf_path}: {str(e)}")
        try:
            os.unlink(temp_path)
        except:
            pass
        return ""

def extract_text_with_macos_native(pdf_path, debug=False):
    """Extract text using macOS native tools (mdimport/mdls)"""
    try:
        # First try mdls to get metadata
        cmd = ["mdls", str(pdf_path)]
        if debug:
            print(f"  macOS: Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        text = ""
        
        # Extract text content from metadata
        if result.returncode == 0:
            metadata = result.stdout
            # Look for kMDItemTextContent
            match = re.search(r'kMDItemTextContent\s*=\s*"(.*)"', metadata, re.DOTALL)
            if match:
                text = match.group(1)
                if debug:
                    print(f"  macOS: Extracted {len(text)} chars from metadata")
        
        # If that didn't work, try textutil
        if len(text.strip()) < 100:
            # Create a temporary file for the text output
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_file:
                temp_path = temp_file.name
            
            cmd = ["textutil", "-convert", "txt", "-output", temp_path, str(pdf_path)]
            if debug:
                print(f"  macOS: Running command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                with open(temp_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                if debug:
                    print(f"  macOS: Extracted {len(text)} chars with textutil")
            
            os.unlink(temp_path)
        
        return text
    except Exception as e:
        if debug:
            print(f"macOS native tools error with {pdf_path}: {str(e)}")
        return ""

def extract_embedded_text(pdf_path, output_path, debug=False):
    """Extract embedded text from a PDF file using multiple methods optimized for macOS"""
    try:
        if debug:
            print(f"\nProcessing {pdf_path.name}...")
        else:
            print(f"Processing {pdf_path.name}...")
        
        # Skip if already processed and not empty
        if output_path.exists() and os.path.getsize(output_path) > 100:
            print(f"Skipping {pdf_path.name} - already processed")
            return True
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path.parent, exist_ok=True)
        
        # Try multiple extraction methods and use the best result
        extraction_results = {}
        
        # Method 1: PyMuPDF (usually fast and reliable)
        if debug:
            print(f"Extracting text from {pdf_path.name} using PyMuPDF...")
        pymupdf_text = extract_text_with_pymupdf(pdf_path, debug)
        extraction_results["pymupdf"] = len(pymupdf_text.strip())
        
        # Method 2: pdfminer.six (good for complex PDFs)
        if debug:
            print(f"Extracting text from {pdf_path.name} using pdfminer.six...")
        pdfminer_text = extract_text_with_pdfminer(pdf_path, debug)
        extraction_results["pdfminer"] = len(pdfminer_text.strip())
        
        # Method 3: pdfplumber (works well on macOS)
        if debug:
            print(f"Extracting text from {pdf_path.name} using pdfplumber...")
        pdfplumber_text = extract_text_with_pdfplumber(pdf_path, debug)
        extraction_results["pdfplumber"] = len(pdfplumber_text.strip())
        
        # Method 4: poppler-utils (pdftotext, good for simple PDFs)
        if debug:
            print(f"Extracting text from {pdf_path.name} using poppler-utils...")
        poppler_text = extract_text_with_poppler(pdf_path, debug)
        extraction_results["poppler"] = len(poppler_text.strip())
        
        # Method 5: pikepdf (good for accessing PDF internals)
        if debug:
            print(f"Extracting text from {pdf_path.name} using pikepdf...")
        pikepdf_text = extract_text_with_pikepdf(pdf_path, debug)
        extraction_results["pikepdf"] = len(pikepdf_text.strip())
        
        # Method 6: qpdf (good for damaged PDFs)
        if debug:
            print(f"Extracting text from {pdf_path.name} using qpdf...")
        qpdf_text = extract_text_with_qpdf(pdf_path, debug)
        extraction_results["qpdf"] = len(qpdf_text.strip())
        
        # Method 7: macOS native tools
        if debug:
            print(f"Extracting text from {pdf_path.name} using macOS native tools...")
        macos_text = extract_text_with_macos_native(pdf_path, debug)
        extraction_results["macos_native"] = len(macos_text.strip())
        
        # Determine which method extracted the most text
        best_method = max(extraction_results, key=extraction_results.get)
        print(f"Best extraction method for {pdf_path.name}: {best_method} ({extraction_results[best_method]} chars)")
        
        # Use the text from the best method
        if best_method == "pymupdf":
            text = pymupdf_text
        elif best_method == "pdfminer":
            text = pdfminer_text
        elif best_method == "pdfplumber":
            text = pdfplumber_text
        elif best_method == "poppler":
            text = poppler_text
        elif best_method == "pikepdf":
            text = pikepdf_text
        elif best_method == "qpdf":
            text = qpdf_text
        else:  # macos_native
            text = macos_text
        
        # If best method yielded minimal text, try combining methods
        if extraction_results[best_method] < 100:
            if debug:
                print(f"Best method yielded minimal text ({extraction_results[best_method]} chars), trying combined approach")
            
            # Combine all non-empty results
            combined_text = ""
            for method_text in [pymupdf_text, pdfminer_text, pdfplumber_text, poppler_text, pikepdf_text, qpdf_text, macos_text]:
                if method_text.strip():
                    combined_text += method_text + "\n\n"
            
            if len(combined_text.strip()) > extraction_results[best_method]:
                if debug:
                    print(f"Combined approach yielded better results: {len(combined_text.strip())} chars")
                text = combined_text
                best_method = "combined"
                extraction_results["combined"] = len(combined_text.strip())
        
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
    parser = argparse.ArgumentParser(description='Extract embedded OCR text from PDF files - Enhanced for macOS')
    parser.add_argument('--input', default='original_files', help='Input directory containing PDF files')
    parser.add_argument('--output', default='extracted_text', help='Output directory for extracted text')
    parser.add_argument('--batch', type=int, default=10, help='Number of files to process in each batch')
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
            if extract_embedded_text(pdf_file, output_path, args.debug):
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
