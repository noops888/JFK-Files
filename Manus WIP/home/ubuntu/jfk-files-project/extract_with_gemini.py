import os
import time
import requests
import base64
import json
from pathlib import Path
import random

# Configuration
ORIGINAL_FILES_DIR = "original_files"
EXTRACTED_TEXT_DIR = "extracted_text"
API_KEY = "YOUR_ACTUAL_API_KEY_HERE"  # Replace with your actual API key

# Create output directory
os.makedirs(EXTRACTED_TEXT_DIR, exist_ok=True)

def process_file(pdf_path):
    """Process a single PDF file with direct API calls"""
    try:
        # Get filename
        filename = pdf_path.stem
        output_path = Path(EXTRACTED_TEXT_DIR) / f"{filename}.txt"
        
        # Skip if already processed
        if output_path.exists():
            print(f"Skipping {filename} - already processed")
            return True
            
        print(f"Processing {filename}...")
        
        # Read PDF file and encode as base64
        with open(pdf_path, 'rb') as f:
            pdf_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Prepare API request
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        data = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "application/pdf",
                                "data": pdf_data
                            }
                        },
                        {
                            "text": "Extract all text from this PDF document. Return only the raw text without any formatting or commentary."
                        }
                    ]
                }
            ],
            "generation_config": {
                "temperature": 0
            }
        }
        
        # Make API request
        response = requests.post(url, headers=headers, json=data) 
        
        # Check if request was successful
        if response.status_code == 200:
            result = response.json()
            
            # Extract text from response
            if 'candidates' in result and len(result['candidates']) > 0:
                text = result['candidates'][0]['content']['parts'][0]['text']
                
                # Save the extracted text
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                print(f"Successfully processed {filename}")
                return True
            else:
                print(f"Error: No text in response for {filename}")
                return False
        else:
            print(f"API Error ({response.status_code}): {response.text}")
            return False
        
    except Exception as e:
        print(f"Error processing {pdf_path.name}: {str(e)}")
        return False

def main():
    # Get list of PDF files
    pdf_dir = Path(ORIGINAL_FILES_DIR)
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    
    print(f"Found {len(pdf_files)} PDF files")
    
    # Process files one by one with rate limiting
    successful = 0
    failed = 0
    
    for i, pdf_file in enumerate(pdf_files):
        # Process the file
        if process_file(pdf_file):
            successful += 1
        else:
            failed += 1
            
        # Add delay between files to avoid rate limits
        if i < len(pdf_files) - 1:  # Skip delay after last file
            delay = random.uniform(2, 5)  # Random delay between 2-5 seconds
            print(f"Waiting {delay:.1f} seconds before next file...")
            time.sleep(delay)
            
        # Add a longer pause every 10 files
        if (i + 1) % 10 == 0 and i < len(pdf_files) - 1:
            pause = 30
            print(f"Processed 10 files. Taking a {pause} second break to avoid rate limits...")
            time.sleep(pause)
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {successful} files")
    print(f"Failed: {failed} files")
    print(f"Success rate: {successful/(successful+failed)*100:.2f}%")

if __name__ == "__main__":
    main()
