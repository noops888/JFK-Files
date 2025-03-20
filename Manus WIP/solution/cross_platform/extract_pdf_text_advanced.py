#!/usr/bin/env python3
"""
JFK Files PDF Extraction Tool - Advanced Image-Based Solution

This script extracts text from JFK Files PDFs that contain only images without proper text layers.
It prioritizes macOS-native tools when available and provides cross-platform alternatives.

Requirements:
- macOS for optimal results (using textutil and mdls)
- poppler-utils (pdfimages, pdftotext)
- tesseract-ocr (optional, for fallback OCR)
- PIL/Pillow (for image processing)

Usage:
python extract_pdf_text_advanced.py [--input INPUT_DIR] [--output OUTPUT_DIR] [--macos_only] [--debug]
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
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Check if running on macOS
IS_MACOS = platform.system() == "Darwin"

# Try to import optional dependencies
try:
    from PIL import Image
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False

# Function to check if tesseract is installed
def is_tesseract_installed():
    try:
        subprocess.run(['tesseract', '--version'], capture_output=True, check=False)
        return True
    except FileNotFoundError:
        return False

# Check if tesseract is installed
HAVE_TESSERACT = is_tesseract_installed()

# Function to check if poppler-utils are installed
def is_poppler_installed():
    try:
        subprocess.run(['pdfimages', '-v'], capture_output=True, check=False)
        return True
    except FileNotFoundError:
        return False

# Check if poppler-utils are installed
HAVE_POPPLER = is_poppler_installed()

def extract_text_with_macos_textutil(pdf_path, debug=False):
    """Extract text using macOS textutil command"""
    if not IS_MACOS:
        if debug:
            print("Not running on macOS, skipping textutil extraction")
        return ""
    
    try:
        # Create a temporary file for the extracted text
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Run textutil to convert PDF to text
        cmd = ["textutil", "-convert", "txt", "-output", temp_path, str(pdf_path)]
        if debug:
            print(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            # Read the extracted text
            with open(temp_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            # Clean up the temporary file
            os.unlink(temp_path)
            
            if debug:
                print(f"Extracted {len(text)} characters with textutil")
            
            return text
        else:
            if debug:
                print(f"textutil failed with return code {result.returncode}")
                print(f"Error: {result.stderr}")
            
            # Clean up the temporary file
            os.unlink(temp_path)
            
            return ""
    except Exception as e:
        if debug:
            print(f"Error using textutil: {str(e)}")
        return ""

def extract_text_with_macos_mdls(pdf_path, debug=False):
    """Extract text using macOS mdls command to get metadata"""
    if not IS_MACOS:
        if debug:
            print("Not running on macOS, skipping mdls extraction")
        return ""
    
    try:
        # Run mdls to get metadata
        cmd = ["mdls", str(pdf_path)]
        if debug:
            print(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            # Extract text content from metadata
            metadata = result.stdout
            
            # Look for kMDItemTextContent
            match = re.search(r'kMDItemTextContent\s*=\s*"(.*)"', metadata, re.DOTALL)
            if match:
                text = match.group(1)
                
                if debug:
                    print(f"Extracted {len(text)} characters with mdls")
                
                return text
            else:
                if debug:
                    print("No text content found in metadata")
                return ""
        else:
            if debug:
                print(f"mdls failed with return code {result.returncode}")
                print(f"Error: {result.stderr}")
            return ""
    except Exception as e:
        if debug:
            print(f"Error using mdls: {str(e)}")
        return ""

def extract_text_with_macos_sips(pdf_path, debug=False):
    """Extract text using macOS sips command to get metadata"""
    if not IS_MACOS:
        if debug:
            print("Not running on macOS, skipping sips extraction")
        return ""
    
    try:
        # Run sips to get metadata
        cmd = ["sips", "-g", "all", str(pdf_path)]
        if debug:
            print(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            # Extract text content from metadata
            metadata = result.stdout
            
            if debug:
                print(f"sips metadata: {metadata}")
            
            # Look for text content in the metadata
            # (sips doesn't typically extract text content, but we're checking just in case)
            return ""
        else:
            if debug:
                print(f"sips failed with return code {result.returncode}")
                print(f"Error: {result.stderr}")
            return ""
    except Exception as e:
        if debug:
            print(f"Error using sips: {str(e)}")
        return ""

def extract_text_with_macos_qlmanage(pdf_path, debug=False):
    """Extract text using macOS qlmanage command to generate preview"""
    if not IS_MACOS:
        if debug:
            print("Not running on macOS, skipping qlmanage extraction")
        return ""
    
    try:
        # Create a temporary directory for the preview
        temp_dir = tempfile.mkdtemp()
        
        # Run qlmanage to generate preview
        cmd = ["qlmanage", "-p", "-o", temp_dir, str(pdf_path)]
        if debug:
            print(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            # Check if a text file was generated
            text_files = list(Path(temp_dir).glob("*.txt"))
            
            if text_files:
                # Read the text file
                with open(text_files[0], 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                
                if debug:
                    print(f"Extracted {len(text)} characters with qlmanage")
                
                # Clean up the temporary directory
                shutil.rmtree(temp_dir)
                
                return text
            else:
                if debug:
                    print("No text file generated by qlmanage")
                
                # Clean up the temporary directory
                shutil.rmtree(temp_dir)
                
                return ""
        else:
            if debug:
                print(f"qlmanage failed with return code {result.returncode}")
                print(f"Error: {result.stderr}")
            
            # Clean up the temporary directory
            shutil.rmtree(temp_dir)
            
            return ""
    except Exception as e:
        if debug:
            print(f"Error using qlmanage: {str(e)}")
        return ""

def extract_text_with_pdftotext(pdf_path, debug=False):
    """Extract text using pdftotext command from poppler-utils"""
    if not HAVE_POPPLER:
        if debug:
            print("poppler-utils not installed, skipping pdftotext extraction")
        return ""
    
    try:
        # Try different pdftotext options
        options = [
            ["-layout"],
            ["-raw"],
            ["-htmlmeta"],
            ["-bbox"],
            ["-layout", "-enc", "UTF-8"],
            ["-raw", "-enc", "UTF-8"],
            ["-layout", "-eol", "unix"],
            ["-raw", "-eol", "unix"],
            ["-layout", "-nopgbrk"],
            ["-raw", "-nopgbrk"]
        ]
        
        best_text = ""
        best_option = None
        
        for option in options:
            cmd = ["pdftotext"] + option + [str(pdf_path), "-"]
            if debug:
                print(f"Running command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                text = result.stdout
                
                if len(text) > len(best_text):
                    best_text = text
                    best_option = option
            else:
                if debug:
                    print(f"pdftotext with options {option} failed with return code {result.returncode}")
                    print(f"Error: {result.stderr}")
        
        if best_option and debug:
            print(f"Best pdftotext option: {' '.join(best_option)}")
            print(f"Extracted {len(best_text)} characters with pdftotext")
        
        return best_text
    except Exception as e:
        if debug:
            print(f"Error using pdftotext: {str(e)}")
        return ""

def extract_images_from_pdf(pdf_path, output_dir, debug=False):
    """Extract images from PDF using pdfimages command from poppler-utils"""
    if not HAVE_POPPLER:
        if debug:
            print("poppler-utils not installed, skipping image extraction")
        return []
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Run pdfimages to extract images
        cmd = ["pdfimages", "-all", str(pdf_path), os.path.join(output_dir, "page")]
        if debug:
            print(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            # Get list of extracted images
            image_files = sorted(Path(output_dir).glob("page-*"))
            
            if debug:
                print(f"Extracted {len(image_files)} images from PDF")
            
            return image_files
        else:
            if debug:
                print(f"pdfimages failed with return code {result.returncode}")
                print(f"Error: {result.stderr}")
            return []
    except Exception as e:
        if debug:
            print(f"Error extracting images: {str(e)}")
        return []

def ocr_image_with_tesseract(image_path, debug=False):
    """OCR an image using tesseract"""
    if not HAVE_TESSERACT:
        if debug:
            print("tesseract not installed, skipping OCR")
        return ""
    
    try:
        # Create a temporary file for the OCR output
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Run tesseract to OCR the image
        cmd = ["tesseract", str(image_path), temp_path.replace('.txt', '')]
        if debug:
            print(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            # Read the OCR output
            with open(temp_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            # Clean up the temporary file
            os.unlink(temp_path)
            
            if debug:
                print(f"OCR extracted {len(text)} characters from {image_path.name}")
            
            return text
        else:
            if debug:
                print(f"tesseract failed with return code {result.returncode}")
                print(f"Error: {result.stderr}")
            
            # Clean up the temporary file
            os.unlink(temp_path)
            
            return ""
    except Exception as e:
        if debug:
            print(f"Error using tesseract: {str(e)}")
        return ""

def process_pdf_file(pdf_path, output_dir, macos_only=False, debug=False):
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
        
        # Try macOS-native methods first if running on macOS
        if IS_MACOS:
            # Method 1: textutil
            if debug:
                print("Trying macOS textutil...")
            textutil_text = extract_text_with_macos_textutil(pdf_path, debug)
            extraction_results["textutil"] = len(textutil_text)
            
            if len(textutil_text) > 100:  # Consider it successful if we got some text
                # Save the extracted text
                with open(output_text_path, 'w', encoding='utf-8') as f:
                    f.write(textutil_text)
                
                # Save extraction metadata
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "filename": pdf_path.name,
                        "extraction_method": "textutil",
                        "characters": len(textutil_text),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }, f, indent=2)
                
                print(f"Successfully extracted {len(textutil_text)} characters from {pdf_path.name} using textutil")
                return True
            
            # Method 2: mdls
            if debug:
                print("Trying macOS mdls...")
            mdls_text = extract_text_with_macos_mdls(pdf_path, debug)
            extraction_results["mdls"] = len(mdls_text)
            
            if len(mdls_text) > 100:  # Consider it successful if we got some text
                # Save the extracted text
                with open(output_text_path, 'w', encoding='utf-8') as f:
                    f.write(mdls_text)
                
                # Save extraction metadata
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "filename": pdf_path.name,
                        "extraction_method": "mdls",
                        "characters": len(mdls_text),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }, f, indent=2)
                
                print(f"Successfully extracted {len(mdls_text)} characters from {pdf_path.name} using mdls")
                return True
            
            # Method 3: qlmanage
            if debug:
                print("Trying macOS qlmanage...")
            qlmanage_text = extract_text_with_macos_qlmanage(pdf_path, debug)
            extraction_results["qlmanage"] = len(qlmanage_text)
            
            if len(qlmanage_text) > 100:  # Consider it successful if we got some text
                # Save the extracted text
                with open(output_text_path, 'w', encoding='utf-8') as f:
                    f.write(qlmanage_text)
                
                # Save extraction metadata
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "filename": pdf_path.name,
                        "extraction_method": "qlmanage",
                        "characters": len(qlmanage_text),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }, f, indent=2)
                
                print(f"Successfully extracted {len(qlmanage_text)} characters from {pdf_path.name} using qlmanage")
                return True
        
        # If macOS-only mode is enabled, skip non-macOS methods
        if macos_only:
            if debug:
                print("macOS-only mode enabled, skipping non-macOS methods")
            
            # Save extraction metadata
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "filename": pdf_path.name,
                    "extraction_results": extraction_results,
                    "error": "No text extracted with macOS-only methods",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }, f, indent=2)
            
            print(f"Warning: Could not extract text from {pdf_path.name} using macOS-only methods")
            return False
        
        # Try pdftotext
        if debug:
            print("Trying pdftotext...")
        pdftotext_text = extract_text_with_pdftotext(pdf_path, debug)
        extraction_results["pdftotext"] = len(pdftotext_text)
        
        if len(pdftotext_text) > 100:  # Consider it successful if we got some text
            # Save the extracted text
            with open(output_text_path, 'w', encoding='utf-8') as f:
                f.write(pdftotext_text)
            
            # Save extraction metadata
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "filename": pdf_path.name,
                    "extraction_method": "pdftotext",
                    "characters": len(pdftotext_text),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }, f, indent=2)
            
            print(f"Successfully extracted {len(pdftotext_text)} characters from {pdf_path.name} using pdftotext")
            return True
        
        # If all direct extraction methods failed, try extracting images and OCR
        if HAVE_TESSERACT and HAVE_POPPLER:
            if debug:
                print("Direct extraction failed, trying image extraction and OCR...")
            
            # Create a temporary directory for extracted images
            temp_dir = tempfile.mkdtemp()
            
            # Extract images from PDF
            image_files = extract_images_from_pdf(pdf_path, temp_dir, debug)
            
            if image_files:
                # OCR each image
                all_text = ""
                
                for image_file in image_files:
                    image_text = ocr_image_with_tesseract(image_file, debug)
                    all_text += image_text + "\n\n"
                
                # Clean up the temporary directory
                shutil.rmtree(temp_dir)
                
                if len(all_text) > 100:  # Consider it successful if we got some text
                    # Save the extracted text
                    with open(output_text_path, 'w', encoding='utf-8') as f:
                        f.write(all_text)
                    
                    # Save extraction metadata
                    with open(output_json_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            "filename": pdf_path.name,
                            "extraction_method": "image_ocr",
                            "characters": len(all_text),
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                        }, f, indent=2)
                    
                    print(f"Successfully extracted {len(all_text)} characters from {pdf_path.name} using image OCR")
                    return True
                else:
                    if debug:
                        print("OCR failed to extract meaningful text")
            else:
                if debug:
                    print("Failed to extract images from PDF")
                
                # Clean up the temporary directory
                shutil.rmtree(temp_dir)
        
        # If all methods failed, save extraction metadata
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
    parser = argparse.ArgumentParser(description='Extract text from JFK Files PDFs')
    parser.add_argument('--input', default='pdf_files', help='Input directory containing PDF files')
    parser.add_argument('--output', default='extracted_text', help='Output directory for extracted text')
    parser.add_argument('--macos_only', action='store_true', help='Use only macOS-native extraction methods')
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
    if IS_MACOS:
        print("Running on macOS - will use native macOS tools for extraction")
    else:
        print("Not running on macOS - will use cross-platform extraction methods")
        if args.macos_only:
            print("Warning: macOS-only mode enabled but not running on macOS")
    
    # Check for required tools
    if not HAVE_POPPLER:
        print("Warning: poppler-utils not installed, some extraction methods will be unavailable")
    
    if not HAVE_TESSERACT:
        print("Warning: tesseract-ocr not installed, OCR fallback will be unavailable")
    
    if not HAVE_PIL:
        print("Warning: PIL/Pillow not installed, some image processing will be unavailable")
    
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
                args.macos_only, 
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
