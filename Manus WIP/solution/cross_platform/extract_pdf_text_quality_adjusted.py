#!/usr/bin/env python3
"""
JFK Files PDF Extraction Tool - Quality-Focused Solution with Adjusted Filtering

This script extracts meaningful text content from PDF files without requiring the Gemini API.
It uses a more lenient approach to text quality filtering while still removing obvious PDF syntax.
It prioritizes macOS-native tools when available but provides cross-platform alternatives.

Requirements:
- pikepdf
- pdfminer.six
- pymupdf (PyMuPDF)
- pdfplumber
- tqdm
- beautifulsoup4

Installation:
pip install pikepdf pdfminer.six pymupdf pdfplumber tqdm beautifulsoup4

Usage:
python extract_pdf_text_quality_adjusted.py [--input INPUT_DIR] [--output OUTPUT_DIR] [--batch BATCH_SIZE] [--debug]
"""

import os
import sys
import argparse
import time
import re
import json
import subprocess
import tempfile
import platform
import io
from pathlib import Path
from tqdm import tqdm
from bs4 import BeautifulSoup

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
    from pdfminer.layout import LAParams
    from pdfminer.pdfdocument import PDFDocument
    from pdfminer.pdfparser import PDFParser
    from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
    from pdfminer.converter import TextConverter
    from pdfminer.pdfpage import PDFPage
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

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("Error: BeautifulSoup library not found. Please install it using:")
    print("pip install beautifulsoup4")
    exit(1)

def is_meaningful_text(text, lenient=True):
    """
    Determine if text contains meaningful content rather than PDF syntax or gibberish
    With adjusted parameters for more lenient filtering
    """
    if not text or len(text.strip()) < 5:  # Reduced minimum length requirement
        return False
    
    # Check for HTML/XML structure and extract text if needed
    if re.search(r'<\/?[a-zA-Z]+[^>]*>', text):
        try:
            soup = BeautifulSoup(text, 'html.parser')
            extracted_text = soup.get_text()
            if extracted_text and len(extracted_text.strip()) > 5:
                return True
        except:
            pass
    
    # Check for common PDF syntax patterns (only filter out obvious syntax)
    pdf_syntax_patterns = [
        r'^(\s*<<.*>>)+\s*$',  # PDF dictionary objects
        r'^(\s*\[\s*.*\]\s*)+$',  # PDF array objects
        r'^\s*stream\s*.*\s*endstream\s*$',  # PDF stream objects
        r'^\s*obj\s*.*\s*endobj\s*$',  # PDF objects
    ]
    
    # Only apply strict filtering if not in lenient mode
    if not lenient:
        for pattern in pdf_syntax_patterns:
            if re.match(pattern, text, re.DOTALL):
                return False
    
    # In lenient mode, accept text with some alphabetic characters
    if lenient:
        # Count alphabetic characters
        alpha_chars = sum(c.isalpha() for c in text)
        total_chars = len(text)
        
        # If at least 5% of characters are alphabetic, consider it potentially meaningful
        if alpha_chars > 0 and alpha_chars / total_chars >= 0.05:
            return True
        
        # Check for presence of common English words as a sign of meaningful content
        common_words = ['the', 'and', 'for', 'that', 'this', 'with', 'from', 'have', 'not']
        text_lower = text.lower()
        for word in common_words:
            if f" {word} " in text_lower:
                return True
    else:
        # More strict checks for non-lenient mode
        total_chars = len(text)
        if total_chars == 0:
            return False
        
        alpha_chars = sum(c.isalpha() for c in text)
        space_chars = sum(c.isspace() for c in text)
        punct_chars = sum(c in '.,;:!?"\'()[]{}' for c in text)
        special_chars = total_chars - alpha_chars - space_chars - punct_chars
        
        # If more than 30% of characters are special characters, likely gibberish
        if special_chars / total_chars > 0.3:
            return False
        
        # If less than 30% of characters are alphabetic, likely gibberish
        if alpha_chars / total_chars < 0.3:
            return False
        
        # Check for reasonable word length distribution
        words = text.split()
        if not words:
            return False
        
        avg_word_length = sum(len(word) for word in words) / len(words)
        # Average English word length is around 4.7 characters
        # If average word length is too far from this, likely gibberish
        if avg_word_length < 2 or avg_word_length > 15:
            return False
    
    return True

def extract_text_from_html(html_text):
    """Extract text content from HTML/XML"""
    try:
        soup = BeautifulSoup(html_text, 'html.parser')
        return soup.get_text()
    except:
        # If BeautifulSoup fails, try simple regex-based extraction
        text = re.sub(r'<[^>]+>', ' ', html_text)
        return text

def clean_text(text):
    """Enhanced cleaning of extracted text with focus on quality"""
    if not text or len(text.strip()) == 0:
        return ""
    
    # Check if text is HTML/XML and extract text content
    if re.search(r'<\/?[a-zA-Z]+[^>]*>', text):
        text = extract_text_from_html(text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common OCR errors
    text = re.sub(r'([a-z])\s+([a-z])', r'\1\2', text)  # Fix split words
    
    # Remove non-printable characters
    text = re.sub(r'[^\x20-\x7E\n]', '', text)
    
    # Normalize line breaks
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove PDF artifacts that might appear in extracted text
    text = re.sub(r'obj\s+endobj', '', text)
    text = re.sub(r'stream\s+endstream', '', text)
    
    # Remove common PDF syntax elements
    text = re.sub(r'/T[a-zA-Z0-9]+\s+', '', text)
    text = re.sub(r'/F[a-zA-Z0-9]+\s+', '', text)
    
    return text.strip()

def extract_text_with_pymupdf_quality(pdf_path, debug=False):
    """Extract quality text using PyMuPDF (fitz) with focus on meaningful content"""
    try:
        all_text = ""
        with fitz.open(pdf_path) as doc:
            # Try to repair document if needed
            if doc.needs_pass or doc.is_encrypted:
                try:
                    # Try with empty password
                    doc.authenticate("")
                except:
                    if debug:
                        print("  PyMuPDF: Document is encrypted and could not be decrypted")
            
            for page_num, page in enumerate(doc):
                if debug:
                    print(f"  PyMuPDF: Processing page {page_num+1}/{len(doc)}")
                
                # Try different extraction methods
                methods = [
                    ("text", page.get_text("text")),
                    ("html", page.get_text("html")),
                    ("dict", page.get_text("dict")),
                    ("json", page.get_text("json")),
                    ("rawdict", page.get_text("rawdict")),
                    ("xhtml", page.get_text("xhtml"))
                ]
                
                # Process HTML formats
                for i, (method_name, content) in enumerate(methods):
                    if method_name in ["html", "xhtml"]:
                        methods[i] = (method_name, extract_text_from_html(content))
                
                # Find the method that extracts the most text
                best_method = max(methods, key=lambda x: len(x[1]))
                method_name, page_text = best_method
                
                if debug:
                    print(f"  PyMuPDF: Best method for page {page_num+1}: {method_name} ({len(page_text)} chars)")
                
                # Try to extract text from annotations as well
                try:
                    annot_text = ""
                    for annot in page.annots():
                        if annot.info.get("content", ""):
                            annot_text += annot.info["content"] + "\n"
                    if annot_text:
                        page_text += "\n" + annot_text
                except:
                    pass
                
                # Add page text to overall text
                all_text += page_text + "\n\n"
        
        return all_text
    except Exception as e:
        if debug:
            print(f"PyMuPDF error with {pdf_path}: {str(e)}")
        return ""

def extract_text_with_pdfminer_quality(pdf_path, debug=False):
    """Extract quality text using pdfminer.six with focus on meaningful content"""
    try:
        # Try multiple approaches with pdfminer
        all_text = ""
        
        # Approach 1: Direct extraction with various parameters
        laparams_options = [
            {"line_margin": 0.5, "char_margin": 2.0, "word_margin": 0.1},
            {"line_margin": 0.3, "char_margin": 1.0, "word_margin": 0.2},
            {"line_margin": 0.2, "char_margin": 0.5, "word_margin": 0.1},
            {"line_margin": 0.1, "char_margin": 0.3, "word_margin": 0.1, "boxes_flow": 0.5},
            {"line_margin": 0.1, "char_margin": 0.3, "word_margin": 0.1, "boxes_flow": -1.0},
            None  # Default parameters
        ]
        
        best_text = ""
        best_params = None
        
        for params in laparams_options:
            try:
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
        
        all_text = best_text
        
        # Approach 2: Low-level extraction with custom converter
        if len(all_text.strip()) < 100:
            try:
                output_string = io.StringIO()
                with open(pdf_path, 'rb') as in_file:
                    parser = PDFParser(in_file)
                    doc = PDFDocument(parser)
                    rsrcmgr = PDFResourceManager()
                    device = TextConverter(rsrcmgr, output_string, laparams=LAParams(
                        line_margin=0.1,
                        char_margin=0.1,
                        word_margin=0.1,
                        boxes_flow=0.5,
                        detect_vertical=True
                    ))
                    interpreter = PDFPageInterpreter(rsrcmgr, device)
                    for page in PDFPage.create_pages(doc):
                        interpreter.process_page(page)
                
                low_level_text = output_string.getvalue()
                if len(low_level_text) > len(all_text):
                    all_text = low_level_text
                    if debug:
                        print(f"  PDFMiner: Low-level extraction yielded better results ({len(low_level_text)} chars)")
            except Exception as e:
                if debug:
                    print(f"  PDFMiner low-level extraction error: {str(e)}")
        
        return all_text
    except Exception as e:
        if debug:
            print(f"PDFMiner error with {pdf_path}: {str(e)}")
        return ""

def extract_text_with_pdfplumber_quality(pdf_path, debug=False):
    """Extract quality text using pdfplumber with focus on meaningful content"""
    try:
        all_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                if debug:
                    print(f"  PDFPlumber: Processing page {i+1}/{len(pdf.pages)}")
                
                # Try different extraction methods
                methods = []
                
                # Standard extraction
                try:
                    standard_text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                    methods.append(("standard", standard_text))
                except Exception as e:
                    if debug:
                        print(f"  PDFPlumber standard extraction error: {str(e)}")
                
                # Try with different tolerances
                try:
                    alt_text = page.extract_text(x_tolerance=5, y_tolerance=5) or ""
                    methods.append(("alt_tolerance", alt_text))
                except Exception as e:
                    if debug:
                        print(f"  PDFPlumber alt tolerance extraction error: {str(e)}")
                
                # Words extraction
                try:
                    words = page.extract_words(x_tolerance=3, y_tolerance=3)
                    words_text = " ".join([word.get("text", "") for word in words])
                    methods.append(("words", words_text))
                except Exception as e:
                    if debug:
                        print(f"  PDFPlumber words extraction error: {str(e)}")
                
                # Words with different settings
                try:
                    words_alt = page.extract_words(x_tolerance=5, y_tolerance=5, keep_blank_chars=True)
                    words_alt_text = " ".join([word.get("text", "") for word in words_alt])
                    methods.append(("words_alt", words_alt_text))
                except Exception as e:
                    if debug:
                        print(f"  PDFPlumber alt words extraction error: {str(e)}")
                
                # Table extraction
                try:
                    tables = page.extract_tables()
                    table_text = "\n".join(["\n".join([" | ".join([str(cell or "") for cell in row]) for row in table]) for table in tables])
                    methods.append(("tables", table_text))
                except Exception as e:
                    if debug:
                        print(f"  PDFPlumber table extraction error: {str(e)}")
                
                # Find the method that extracts the most text
                if methods:
                    best_method = max(methods, key=lambda x: len(x[1]))
                    method_name, page_text = best_method
                    
                    if debug:
                        print(f"  PDFPlumber: Best method for page {i+1}: {method_name} ({len(page_text)} chars)")
                    
                    all_text += page_text + "\n\n"
        
        return all_text
    except Exception as e:
        if debug:
            print(f"PDFPlumber error with {pdf_path}: {str(e)}")
        return ""

def extract_text_with_poppler_quality(pdf_path, debug=False):
    """Extract quality text using poppler-utils (pdftotext) with focus on meaningful content"""
    try:
        # Check if pdftotext is available
        try:
            subprocess.run(['pdftotext', '-v'], capture_output=True, check=False)
        except FileNotFoundError:
            if debug:
                print("pdftotext not found, skipping poppler extraction")
            return ""
        
        # Try multiple extraction modes and use the best one
        extraction_modes = [
            ['-layout'],       # Maintain original layout
            ['-raw'],          # Raw text extraction
            ['-htmlmeta'],     # HTML with metadata
            ['-bbox'],         # Include bounding box information
            ['-layout', '-enc', 'UTF-8'],  # Layout with UTF-8 encoding
            ['-raw', '-enc', 'UTF-8'],     # Raw with UTF-8 encoding
            ['-layout', '-eol', 'unix'],   # Layout with unix line endings
            ['-raw', '-eol', 'unix'],      # Raw with unix line endings
            ['-layout', '-nopgbrk'],       # Layout without page breaks
            ['-raw', '-nopgbrk']           # Raw without page breaks
        ]
        
        best_text = ""
        best_mode = None
        
        for mode in extraction_modes:
            try:
                cmd = ['pdftotext'] + mode + [str(pdf_path), '-']
                if debug:
                    print(f"  Poppler: Trying {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False
                )
                
                if result.returncode == 0:
                    current_text = result.stdout
                    
                    # Check if the text is HTML and extract content if needed
                    if re.search(r'<\/?[a-zA-Z]+[^>]*>', current_text):
                        current_text = extract_text_from_html(current_text)
                    
                    if len(current_text) > len(best_text):
                        best_text = current_text
                        best_mode = mode
            except Exception as e:
                if debug:
                    print(f"  Poppler error with mode {mode}: {str(e)}")
        
        if debug and best_mode:
            print(f"  Poppler: Best mode: {' '.join(best_mode)} ({len(best_text)} chars)")
        
        return best_text
    except Exception as e:
        if debug:
            print(f"Poppler error with {pdf_path}: {str(e)}")
        return ""

def extract_text_with_pikepdf_quality(pdf_path, debug=False):
    """Extract quality text using pikepdf with focus on meaningful content"""
    try:
        all_text = ""
        with pikepdf.open(pdf_path) as pdf:
            # Extract text from each page
            for page_num, page in enumerate(pdf.pages):
                if debug:
                    print(f"  pikepdf: Processing page {page_num+1}/{len(pdf.pages)}")
                
                page_text = ""
                
                # Method 1: Extract from content streams
                if '/Contents' in page:
                    content = page['/Contents']
                    content_text = ""
                    
                    if isinstance(content, pikepdf.Array):
                        for item in content:
                            try:
                                stream_text = item.read_bytes().decode('utf-8', errors='ignore')
                                content_text += stream_text + "\n"
                            except:
                                pass
                    else:
                        try:
                            stream_text = content.read_bytes().decode('utf-8', errors='ignore')
                            content_text += stream_text + "\n"
                        except:
                            pass
                    
                    # Extract text content from PDF operators
                    # Look for text showing operators like TJ, Tj, ', ", etc.
                    text_matches = re.findall(r'\((.*?)\)\s*Tj|\[(.*?)\]\s*TJ', content_text)
                    for match in text_matches:
                        for group in match:
                            if group:
                                # Decode PDF string escapes
                                decoded = group.replace('\\(', '(').replace('\\)', ')').replace('\\\\', '\\')
                                page_text += decoded + " "
                    
                    # If we didn't find any text with regex, use the raw content
                    if not page_text and content_text:
                        page_text = content_text
                
                # Method 2: Extract from annotations
                if '/Annots' in page:
                    annots = page['/Annots']
                    if isinstance(annots, pikepdf.Array):
                        for annot in annots:
                            if '/Contents' in annot:
                                try:
                                    annot_text = str(annot['/Contents'])
                                    page_text += annot_text + "\n"
                                except:
                                    pass
                            if '/RC' in annot:  # Rich text content
                                try:
                                    rich_text = str(annot['/RC'])
                                    # Remove XML/HTML tags
                                    rich_text = extract_text_from_html(rich_text)
                                    page_text += rich_text + "\n"
                                except:
                                    pass
                
                # Add the page text to the overall text
                all_text += page_text + "\n\n"
        
        # Post-process the text to clean up PDF operators and syntax
        all_text = re.sub(r'BT\s+ET', ' ', all_text)  # Remove begin/end text markers
        all_text = re.sub(r'Tm\s+', ' ', all_text)    # Remove text matrix operators
        all_text = re.sub(r'Td\s+', ' ', all_text)    # Remove text positioning operators
        all_text = re.sub(r'Tf\s+', ' ', all_text)    # Remove font operators
        all_text = re.sub(r'[\d.]+ [\d.]+ [\d.]+ [\d.]+ [\d.]+ [\d.]+ cm', ' ', all_text)  # Remove transformation matrices
        
        return all_text
    except Exception as e:
        if debug:
            print(f"pikepdf error with {pdf_path}: {str(e)}")
        return ""

def extract_text_with_macos_native_quality(pdf_path, debug=False):
    """Extract quality text using macOS native tools with focus on meaningful content"""
    # Only run on macOS
    if platform.system() != "Darwin":
        if debug:
            print("Not running on macOS, skipping macOS native extraction")
        return ""
    
    try:
        # Method 1: Try mdls to get metadata
        cmd = ["mdls", str(pdf_path)]
        if debug:
            print(f"  macOS: Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        # Extract text content from metadata
        if result.returncode == 0:
            metadata = result.stdout
            # Look for kMDItemTextContent
            match = re.search(r'kMDItemTextContent\s*=\s*"(.*)"', metadata, re.DOTALL)
            if match:
                text = match.group(1)
                if debug:
                    print(f"  macOS: Extracted {len(text)} chars from metadata")
                return text
        
        # Method 2: Try textutil
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
        
        os.unlink(temp_path)
        return ""
    except Exception as e:
        if debug:
            print(f"macOS native tools error with {pdf_path}: {str(e)}")
        return ""

def extract_raw_text_fallback(pdf_path, debug=False):
    """
    Extract raw text from PDF without quality filtering as a fallback method
    """
    try:
        # Try to extract any text content, even if it might be PDF syntax
        all_text = ""
        
        # Method 1: Use PyMuPDF to extract raw text
        try:
            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate(doc):
                    # Get raw text without filtering
                    raw_text = page.get_text("text")
                    all_text += raw_text + "\n\n"
        except Exception as e:
            if debug:
                print(f"Raw PyMuPDF extraction error: {str(e)}")
        
        # Method 2: Use pikepdf to extract raw content streams
        if not all_text:
            try:
                with pikepdf.open(pdf_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        if '/Contents' in page:
                            content = page['/Contents']
                            if isinstance(content, pikepdf.Array):
                                for item in content:
                                    try:
                                        stream_text = item.read_bytes().decode('utf-8', errors='ignore')
                                        all_text += stream_text + "\n"
                                    except:
                                        pass
                            else:
                                try:
                                    stream_text = content.read_bytes().decode('utf-8', errors='ignore')
                                    all_text += stream_text + "\n"
                                except:
                                    pass
            except Exception as e:
                if debug:
                    print(f"Raw pikepdf extraction error: {str(e)}")
        
        # Method 3: Use pdftotext with raw option
        if not all_text:
            try:
                cmd = ['pdftotext', '-raw', str(pdf_path), '-']
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False
                )
                if result.returncode == 0:
                    all_text = result.stdout
            except Exception as e:
                if debug:
                    print(f"Raw pdftotext extraction error: {str(e)}")
        
        # Extract any text-like content from the raw text
        if all_text:
            # Extract anything that looks like words (sequences of letters)
            words = re.findall(r'[a-zA-Z]{2,}', all_text)
            if words:
                return " ".join(words)
        
        return all_text
    except Exception as e:
        if debug:
            print(f"Raw text fallback error with {pdf_path}: {str(e)}")
        return ""

def extract_embedded_text_quality(pdf_path, output_path, debug=False):
    """Extract quality text from a PDF file using multiple methods with adjusted filtering"""
    try:
        if debug:
            print(f"\nProcessing {pdf_path.name}...")
        else:
            print(f"Processing {pdf_path.name}...")
        
        # Skip if already processed and not empty
        if output_path.exists() and os.path.getsize(output_path) > 100:
            # Verify that the existing output contains meaningful text
            with open(output_path, 'r', encoding='utf-8', errors='ignore') as f:
                existing_text = f.read()
            if is_meaningful_text(existing_text, lenient=True):
                print(f"Skipping {pdf_path.name} - already processed with meaningful text")
                return True
            else:
                print(f"Reprocessing {pdf_path.name} - existing output lacks meaningful text")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path.parent, exist_ok=True)
        
        # Try multiple extraction methods and use the first that yields meaningful text
        extraction_results = {}
        
        # Method 1: macOS native tools (only on macOS)
        if platform.system() == "Darwin":
            if debug:
                print(f"Extracting text from {pdf_path.name} using macOS native tools...")
            macos_text = extract_text_with_macos_native_quality(pdf_path, debug)
            extraction_results["macos_native"] = len(macos_text.strip())
            
            # If macOS native tools yield meaningful text, use it immediately
            if is_meaningful_text(macos_text, lenient=True):
                cleaned_text = clean_text(macos_text)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_text)
                
                # Save extraction metadata
                metadata_path = output_path.with_suffix('.json')
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "filename": pdf_path.name,
                        "extraction_results": {"macos_native": len(cleaned_text)},
                        "best_method": "macos_native",
                        "extracted_chars": len(cleaned_text),
                        "is_meaningful": True,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }, f, indent=2)
                
                print(f"Successfully extracted {len(cleaned_text)} characters of meaningful text from {pdf_path.name} using macOS native tools")
                return True
        
        # Method 2: PyMuPDF
        if debug:
            print(f"Extracting text from {pdf_path.name} using PyMuPDF...")
        pymupdf_text = extract_text_with_pymupdf_quality(pdf_path, debug)
        extraction_results["pymupdf"] = len(pymupdf_text.strip())
        
        # If PyMuPDF yields meaningful text, use it
        if is_meaningful_text(pymupdf_text, lenient=True):
            cleaned_text = clean_text(pymupdf_text)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            
            # Save extraction metadata
            metadata_path = output_path.with_suffix('.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "filename": pdf_path.name,
                    "extraction_results": extraction_results,
                    "best_method": "pymupdf",
                    "extracted_chars": len(cleaned_text),
                    "is_meaningful": True,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }, f, indent=2)
            
            print(f"Successfully extracted {len(cleaned_text)} characters of meaningful text from {pdf_path.name} using PyMuPDF")
            return True
        
        # Method 3: pdfminer.six
        if debug:
            print(f"Extracting text from {pdf_path.name} using pdfminer.six...")
        pdfminer_text = extract_text_with_pdfminer_quality(pdf_path, debug)
        extraction_results["pdfminer"] = len(pdfminer_text.strip())
        
        # If pdfminer yields meaningful text, use it
        if is_meaningful_text(pdfminer_text, lenient=True):
            cleaned_text = clean_text(pdfminer_text)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            
            # Save extraction metadata
            metadata_path = output_path.with_suffix('.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "filename": pdf_path.name,
                    "extraction_results": extraction_results,
                    "best_method": "pdfminer",
                    "extracted_chars": len(cleaned_text),
                    "is_meaningful": True,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }, f, indent=2)
            
            print(f"Successfully extracted {len(cleaned_text)} characters of meaningful text from {pdf_path.name} using pdfminer.six")
            return True
        
        # Method 4: pdfplumber
        if debug:
            print(f"Extracting text from {pdf_path.name} using pdfplumber...")
        pdfplumber_text = extract_text_with_pdfplumber_quality(pdf_path, debug)
        extraction_results["pdfplumber"] = len(pdfplumber_text.strip())
        
        # If pdfplumber yields meaningful text, use it
        if is_meaningful_text(pdfplumber_text, lenient=True):
            cleaned_text = clean_text(pdfplumber_text)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            
            # Save extraction metadata
            metadata_path = output_path.with_suffix('.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "filename": pdf_path.name,
                    "extraction_results": extraction_results,
                    "best_method": "pdfplumber",
                    "extracted_chars": len(cleaned_text),
                    "is_meaningful": True,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }, f, indent=2)
            
            print(f"Successfully extracted {len(cleaned_text)} characters of meaningful text from {pdf_path.name} using pdfplumber")
            return True
        
        # Method 5: poppler-utils
        if debug:
            print(f"Extracting text from {pdf_path.name} using poppler-utils...")
        poppler_text = extract_text_with_poppler_quality(pdf_path, debug)
        extraction_results["poppler"] = len(poppler_text.strip())
        
        # If poppler yields meaningful text, use it
        if is_meaningful_text(poppler_text, lenient=True):
            cleaned_text = clean_text(poppler_text)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            
            # Save extraction metadata
            metadata_path = output_path.with_suffix('.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "filename": pdf_path.name,
                    "extraction_results": extraction_results,
                    "best_method": "poppler",
                    "extracted_chars": len(cleaned_text),
                    "is_meaningful": True,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }, f, indent=2)
            
            print(f"Successfully extracted {len(cleaned_text)} characters of meaningful text from {pdf_path.name} using poppler-utils")
            return True
        
        # Method 6: pikepdf
        if debug:
            print(f"Extracting text from {pdf_path.name} using pikepdf...")
        pikepdf_text = extract_text_with_pikepdf_quality(pdf_path, debug)
        extraction_results["pikepdf"] = len(pikepdf_text.strip())
        
        # If pikepdf yields meaningful text, use it
        if is_meaningful_text(pikepdf_text, lenient=True):
            cleaned_text = clean_text(pikepdf_text)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            
            # Save extraction metadata
            metadata_path = output_path.with_suffix('.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "filename": pdf_path.name,
                    "extraction_results": extraction_results,
                    "best_method": "pikepdf",
                    "extracted_chars": len(cleaned_text),
                    "is_meaningful": True,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }, f, indent=2)
            
            print(f"Successfully extracted {len(cleaned_text)} characters of meaningful text from {pdf_path.name} using pikepdf")
            return True
        
        # Fallback: Try raw text extraction without quality filtering
        if debug:
            print(f"No meaningful text found, trying raw text extraction as fallback...")
        
        raw_text = extract_raw_text_fallback(pdf_path, debug)
        extraction_results["raw_fallback"] = len(raw_text.strip())
        
        if raw_text and len(raw_text.strip()) > 0:
            cleaned_text = clean_text(raw_text)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            
            # Save extraction metadata
            metadata_path = output_path.with_suffix('.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "filename": pdf_path.name,
                    "extraction_results": extraction_results,
                    "best_method": "raw_fallback",
                    "extracted_chars": len(cleaned_text),
                    "is_meaningful": False,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }, f, indent=2)
            
            print(f"Warning: Could not extract meaningful text from {pdf_path.name}. Using raw fallback method with {len(cleaned_text)} characters.")
            return True
        
        # If no method yields any text, use the method that extracts the most text
        best_method = max(extraction_results, key=extraction_results.get) if extraction_results else None
        
        if best_method and extraction_results[best_method] > 0:
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
            elif best_method == "macos_native":
                text = macos_text
            else:
                text = ""
            
            cleaned_text = clean_text(text)
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
                    "is_meaningful": False,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }, f, indent=2)
            
            print(f"Warning: Could not extract meaningful text from {pdf_path.name}. Using best available method ({best_method}) with {len(cleaned_text)} characters.")
            return True
        else:
            print(f"Error: Could not extract any text from {pdf_path.name} using any method.")
            return False
    
    except Exception as e:
        print(f"Error processing {pdf_path.name}: {str(e)}")
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract quality text from PDF files with adjusted filtering')
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
    
    # Print system information
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    
    # Check if running on macOS
    if platform.system() == "Darwin":
        print("Running on macOS - will use native macOS tools for extraction")
    else:
        print("Not running on macOS - will use cross-platform extraction methods")
    
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
            if extract_embedded_text_quality(pdf_file, output_path, args.debug):
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
