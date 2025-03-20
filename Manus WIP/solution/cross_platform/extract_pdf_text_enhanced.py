#!/usr/bin/env python3
"""
JFK Files PDF Extraction Tool - Enhanced Cross-Platform Solution

This script extracts embedded OCR text from PDF files without requiring the Gemini API.
It uses multiple specialized PDF libraries to access the text layer that already exists
in the PDFs rather than performing unnecessary OCR. It prioritizes macOS-native tools
when available but provides cross-platform alternatives with enhanced extraction methods.

Requirements:
- pikepdf
- pdfminer.six
- pymupdf (PyMuPDF)
- pdfplumber
- tqdm

Installation:
pip install pikepdf pdfminer.six pymupdf pdfplumber tqdm

Usage:
python extract_pdf_text_enhanced.py [--input INPUT_DIR] [--output OUTPUT_DIR] [--batch BATCH_SIZE] [--debug]
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

def clean_text(text):
    """Enhanced cleaning of extracted text"""
    if not text or len(text.strip()) == 0:
        return ""
        
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

def extract_text_with_pymupdf(pdf_path, debug=False):
    """Enhanced text extraction using PyMuPDF (fitz) with improved options"""
    try:
        text = ""
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
                
                text += page_text + "\n\n"
        
        return text
    except Exception as e:
        if debug:
            print(f"PyMuPDF error with {pdf_path}: {str(e)}")
        return ""

def extract_text_with_pdfminer_enhanced(pdf_path, debug=False):
    """Enhanced text extraction using pdfminer.six with custom parameters"""
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

def extract_text_with_pikepdf_enhanced(pdf_path, debug=False):
    """Enhanced text extraction using pikepdf with improved content extraction"""
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
                
                # Method 2: Extract from text-related dictionaries
                if '/Resources' in page and '/Font' in page['/Resources']:
                    # This page has fonts, which suggests it has text
                    if not page_text:
                        # If we haven't extracted text yet, try the raw content approach
                        if '/Contents' in page:
                            content = page['/Contents']
                            if isinstance(content, pikepdf.Array):
                                for item in content:
                                    try:
                                        page_text += item.read_bytes().decode('utf-8', errors='ignore')
                                    except:
                                        pass
                            else:
                                try:
                                    page_text += content.read_bytes().decode('utf-8', errors='ignore')
                                except:
                                    pass
                
                # Method 3: Extract from annotations
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
                                    rich_text = re.sub(r'<[^>]+>', ' ', rich_text)
                                    page_text += rich_text + "\n"
                                except:
                                    pass
                
                # Method 4: Look for text in XObject forms
                if '/Resources' in page and '/XObject' in page['/Resources']:
                    try:
                        xobjects = page['/Resources']['/XObject']
                        if isinstance(xobjects, pikepdf.Dictionary):
                            for key, xobject in xobjects.items():
                                if xobject.get('/Subtype') == '/Form':
                                    if '/Contents' in xobject:
                                        try:
                                            xobject_text = xobject['/Contents'].read_bytes().decode('utf-8', errors='ignore')
                                            page_text += xobject_text + "\n"
                                        except:
                                            pass
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

def extract_text_with_poppler_enhanced(pdf_path, debug=False):
    """Enhanced text extraction using poppler-utils (pdftotext) with multiple options"""
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

def extract_text_with_macos_native(pdf_path, debug=False):
    """Extract text using macOS native tools (mdimport/mdls/textutil)"""
    # Only run on macOS
    if platform.system() != "Darwin":
        if debug:
            print("Not running on macOS, skipping macOS native extraction")
        return ""
    
    try:
        all_text = ""
        
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
                all_text = match.group(1)
                if debug:
                    print(f"  macOS: Extracted {len(all_text)} chars from metadata")
        
        # Method 2: If that didn't work, try textutil
        if len(all_text.strip()) < 100:
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
                    textutil_text = f.read()
                if len(textutil_text) > len(all_text):
                    all_text = textutil_text
                    if debug:
                        print(f"  macOS: Extracted {len(all_text)} chars with textutil")
            
            os.unlink(temp_path)
        
        # Method 3: Try sips to extract text (another macOS tool)
        if len(all_text.strip()) < 100:
            cmd = ["sips", "-g", "allxml", str(pdf_path)]
            if debug:
                print(f"  macOS: Running command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                sips_output = result.stdout
                # Extract text content from XML
                text_matches = re.findall(r'<text>(.*?)</text>', sips_output, re.DOTALL)
                sips_text = "\n".join(text_matches)
                if len(sips_text) > len(all_text):
                    all_text = sips_text
                    if debug:
                        print(f"  macOS: Extracted {len(all_text)} chars with sips")
        
        # Method 4: Try Spotlight indexing directly
        if len(all_text.strip()) < 100:
            cmd = ["mdimport", "-d2", str(pdf_path)]
            if debug:
                print(f"  macOS: Running command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                mdimport_output = result.stdout
                # Look for text content in the debug output
                text_matches = re.findall(r'kMDItemTextContent = "(.*?)"', mdimport_output, re.DOTALL)
                if text_matches:
                    mdimport_text = text_matches[0]
                    if len(mdimport_text) > len(all_text):
                        all_text = mdimport_text
                        if debug:
                            print(f"  macOS: Extracted {len(all_text)} chars with mdimport")
        
        return all_text
    except Exception as e:
        if debug:
            print(f"macOS native tools error with {pdf_path}: {str(e)}")
        return ""

def extract_text_with_qpdf_enhanced(pdf_path, debug=False):
    """Enhanced text extraction after repairing PDF with qpdf"""
    try:
        # Check if qpdf is available
        try:
            subprocess.run(['qpdf', '--version'], capture_output=True, check=False)
        except FileNotFoundError:
            if debug:
                print("qpdf not found, skipping qpdf repair and extraction")
            return ""
        
        # Create a temporary file for the repaired PDF
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Try multiple repair options
        repair_options = [
            ["--replace-input"],
            ["--replace-input", "--decrypt"],
            ["--replace-input", "--linearize"],
            ["--replace-input", "--normalize-content=y"],
            ["--replace-input", "--compress-streams=n"],
            ["--replace-input", "--decrypt", "--compress-streams=n"]
        ]
        
        best_text = ""
        best_option = None
        
        for options in repair_options:
            try:
                # Create a new temp file for each attempt
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as option_temp:
                    option_temp_path = option_temp.name
                
                cmd = ["qpdf"] + options + [str(pdf_path), option_temp_path]
                if debug:
                    print(f"  QPDF: Running command: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False
                )
                
                if result.returncode == 0 or result.returncode == 3:  # 3 means warnings, but still produced output
                    # Try to extract text from the repaired PDF using multiple methods
                    pikepdf_text = extract_text_with_pikepdf_enhanced(option_temp_path, debug)
                    pymupdf_text = extract_text_with_pymupdf(option_temp_path, debug)
                    
                    # Use the method that extracted the most text
                    current_text = pikepdf_text if len(pikepdf_text) > len(pymupdf_text) else pymupdf_text
                    
                    if len(current_text) > len(best_text):
                        best_text = current_text
                        best_option = options
                
                # Clean up the temp file
                os.unlink(option_temp_path)
            
            except Exception as e:
                if debug:
                    print(f"  QPDF error with options {options}: {str(e)}")
        
        if debug and best_option:
            print(f"  QPDF: Best repair options: {' '.join(best_option)} ({len(best_text)} chars)")
        
        return best_text
    except Exception as e:
        if debug:
            print(f"QPDF error with {pdf_path}: {str(e)}")
        return ""

def extract_embedded_text(pdf_path, output_path, debug=False):
    """Extract embedded text from a PDF file using multiple enhanced methods"""
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
            print(f"Extracting text from {pdf_path.name} using enhanced pdfminer.six...")
        pdfminer_text = extract_text_with_pdfminer_enhanced(pdf_path, debug)
        extraction_results["pdfminer"] = len(pdfminer_text.strip())
        
        # Method 3: pdfplumber (good for tables and structured content)
        if debug:
            print(f"Extracting text from {pdf_path.name} using pdfplumber...")
        pdfplumber_text = extract_text_with_pdfplumber(pdf_path, debug)
        extraction_results["pdfplumber"] = len(pdfplumber_text.strip())
        
        # Method 4: poppler-utils (pdftotext, good for simple PDFs)
        if debug:
            print(f"Extracting text from {pdf_path.name} using enhanced poppler-utils...")
        poppler_text = extract_text_with_poppler_enhanced(pdf_path, debug)
        extraction_results["poppler"] = len(poppler_text.strip())
        
        # Method 5: pikepdf (good for accessing PDF internals)
        if debug:
            print(f"Extracting text from {pdf_path.name} using enhanced pikepdf...")
        pikepdf_text = extract_text_with_pikepdf_enhanced(pdf_path, debug)
        extraction_results["pikepdf"] = len(pikepdf_text.strip())
        
        # Method 6: qpdf (repair and extract)
        if debug:
            print(f"Extracting text from {pdf_path.name} using enhanced qpdf repair...")
        qpdf_text = extract_text_with_qpdf_enhanced(pdf_path, debug)
        extraction_results["qpdf"] = len(qpdf_text.strip())
        
        # Method 7: macOS native tools (only on macOS)
        if platform.system() == "Darwin":
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
        elif best_method == "macos_native":
            text = macos_text
        else:
            text = ""
        
        # If best method didn't extract much text, try combining methods
        if extraction_results[best_method] < 1000:
            if debug:
                print(f"Best method only extracted {extraction_results[best_method]} chars, trying combined approach...")
            
            # Combine all non-empty results
            combined_text = ""
            if extraction_results["pymupdf"] > 0:
                combined_text += pymupdf_text + "\n\n"
            if extraction_results["pdfminer"] > 0:
                combined_text += pdfminer_text + "\n\n"
            if extraction_results["pdfplumber"] > 0:
                combined_text += pdfplumber_text + "\n\n"
            if extraction_results["poppler"] > 0:
                combined_text += poppler_text + "\n\n"
            if extraction_results["pikepdf"] > 0:
                combined_text += pikepdf_text + "\n\n"
            if extraction_results["qpdf"] > 0:
                combined_text += qpdf_text + "\n\n"
            if "macos_native" in extraction_results and extraction_results["macos_native"] > 0:
                combined_text += macos_text + "\n\n"
            
            # If combined text is longer, use it instead
            if len(combined_text.strip()) > extraction_results[best_method]:
                text = combined_text
                best_method = "combined"
                extraction_results["combined"] = len(combined_text.strip())
                print(f"Using combined approach: {extraction_results['combined']} chars")
        
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

def extract_text_with_pdfplumber(pdf_path, debug=False):
    """Extract text using pdfplumber with enhanced options"""
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

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract embedded OCR text from PDF files')
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
