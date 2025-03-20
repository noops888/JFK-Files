import os
import subprocess
from pathlib import Path
from tqdm import tqdm

# Configuration
ORIGINAL_FILES_DIR = "original_files"
EXTRACTED_TEXT_DIR = "extracted_text"

# Create output directory
os.makedirs(EXTRACTED_TEXT_DIR, exist_ok=True)

def extract_text_with_pdfminer(pdf_path):
    """Extract text using pdfminer's pdf2txt.py tool"""
    try:
        output_path = Path(EXTRACTED_TEXT_DIR) / f"{pdf_path.stem}.txt"
        
        # Run pdf2txt.py command
        result = subprocess.run(
            ['pdf2txt.py', '-o', str(output_path), str(pdf_path)],
            capture_output=True,
            text=True
        )
        
        # Check if the output file was created and has content
        if output_path.exists() and os.path.getsize(output_path) > 0:
            print(f"Successfully extracted text from {pdf_path.name}")
            return True
        else:
            print(f"Failed to extract text from {pdf_path.name}")
            return False
            
    except Exception as e:
        print(f"Error processing {pdf_path.name}: {str(e)}")
        return False

def main():
    # Get list of PDF files
    pdf_dir = Path(ORIGINAL_FILES_DIR)
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    
    # Limit to 10 files for testing
    pdf_files = pdf_files[:10]
    
    print(f"Found {len(pdf_files)} PDF files")
    
    # Process files
    successful = 0
    failed = 0
    
    for pdf_file in tqdm(pdf_files):
        if extract_text_with_pdfminer(pdf_file):
            successful += 1
        else:
            failed += 1
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful} files")
    print(f"Failed: {failed} files")
    print(f"Success rate: {successful/len(pdf_files)*100:.2f}%")

if __name__ == "__main__":
    main()
