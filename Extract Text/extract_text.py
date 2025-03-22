import os
import subprocess
from pathlib import Path
import time
from tqdm import tqdm

# Configuration
ORIGINAL_FILES_DIR = "original_files"
EXTRACTED_TEXT_DIR = "extracted_text"
MIN_TEXT_LENGTH = 10  # Minimum number of characters to consider a file successfully processed

def extract_text_from_pdf(pdf_path, output_path):
    """Extract text from a PDF file using macOS textutil"""
    try:
        # Convert PDF to RTF using textutil
        rtf_path = output_path.with_suffix('.rtf')
        cmd = ['textutil', '-convert', 'rtf', '-output', str(rtf_path), str(pdf_path)]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"\nError converting {pdf_path.name}:")
            print(result.stderr)
            return False
            
        # Convert RTF to plain text
        txt_cmd = ['textutil', '-convert', 'txt', '-output', str(output_path), str(rtf_path)]
        txt_result = subprocess.run(txt_cmd, capture_output=True, text=True)
        
        # Clean up the RTF file
        rtf_path.unlink()
        
        if txt_result.returncode != 0:
            print(f"\nError converting RTF to text for {pdf_path.name}:")
            print(txt_result.stderr)
            return False
            
        # Verify the output
        if not output_path.exists():
            print(f"\nError: Failed to create output file {output_path.name}")
            return False
            
        file_size = output_path.stat().st_size
        if file_size < MIN_TEXT_LENGTH:
            print(f"\nError: Output file {output_path.name} is too small ({file_size} bytes)")
            output_path.unlink()
            return False
            
        return True
        
    except Exception as e:
        print(f"\nError processing {pdf_path.name}: {str(e)}")
        return False

def main():
    # Create output directory
    os.makedirs(EXTRACTED_TEXT_DIR, exist_ok=True)
    
    # Get list of PDF files
    pdf_dir = Path(ORIGINAL_FILES_DIR)
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    
    if not pdf_files:
        print(f"Error: No PDF files found in {ORIGINAL_FILES_DIR}")
        return
    
    print(f"Found {len(pdf_files)} PDF files")
    
    # Process files with progress bar
    successful = 0
    failed = 0
    skipped = 0
    
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        output_path = Path(EXTRACTED_TEXT_DIR) / f"{pdf_file.stem}.txt"
        
        # Skip if already processed and valid
        if output_path.exists() and output_path.stat().st_size >= MIN_TEXT_LENGTH:
            skipped += 1
            continue
        
        # Process the file
        if extract_text_from_pdf(pdf_file, output_path):
            successful += 1
        else:
            failed += 1
            # Remove empty or invalid output file
            if output_path.exists():
                output_path.unlink()
        
        # Small delay to prevent system overload
        time.sleep(0.1)
    
    # Print results
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful} files")
    print(f"Failed: {failed} files")
    print(f"Skipped (already processed): {skipped} files")
    if successful + failed > 0:
        print(f"Success rate: {successful/(successful+failed)*100:.2f}%")

if __name__ == "__main__":
    main() 