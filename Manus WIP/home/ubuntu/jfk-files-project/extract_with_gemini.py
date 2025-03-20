import os
import time
from google import genai
from google.genai import types
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def convert_pdf_to_text(pdf_path, retries=3, initial_delay=1):
    try:
        # Extract filename without extension
        filename = Path(pdf_path).stem
        output_path = Path("extracted_text") / f"{filename}.txt"
        
        # Skip if already converted
        if output_path.exists():
            print(f"Skipping {filename} - already converted")
            return
        
        # Initialize Gemini API client
        client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY"),
        )
        
        for attempt in range(retries):
            try:
                # Upload the PDF file
                uploaded_file = client.files.upload(file=str(pdf_path))
                
                # Use Gemini Flash model
                model = "gemini-2.0-flash"
                
                # Create the prompt
                contents = [
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_uri(
                                file_uri=uploaded_file.uri,
                                mime_type=uploaded_file.mime_type,
                            ),
                            types.Part.from_text(text="Extract all text from this PDF document. Return only the raw text without any formatting or commentary."),
                        ],
                    ),
                ]
                
                # Configure generation parameters
                generate_content_config = types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="text/plain",
                )
                
                # Create output directory if it doesn't exist
                os.makedirs(output_path.parent, exist_ok=True)
                
                # Process and collect the response
                full_response = ""
                for chunk in client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=generate_content_config,
                ):
                    full_response += chunk.text
                
                # Only write if response is not empty
                if full_response.strip():
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(full_response)
                    print(f"Converted {filename}")
                else:
                    print(f"Skipping {filename} - empty response")
                    raise Exception("Empty response from Gemini")
                
                return
                
            except Exception as e:
                if attempt < retries - 1:
                    delay = initial_delay * (2 ** attempt)  # Exponential backoff
                    time.sleep(delay)
                    print(f"Retrying {filename} after error: {e}")
                else:
                    print(f"Failed to process {filename} after {retries} attempts: {e}")
                    raise
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")

def main():
    # Get list of PDF files
    pdf_dir = Path("original_files")
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    
    # Create output directory
    os.makedirs("extracted_text", exist_ok=True)
    
    # Process files in parallel with ThreadPoolExecutor
    # Adjust max_workers based on your system capabilities and API rate limits
    with ThreadPoolExecutor(max_workers=5) as executor:
        list(tqdm(executor.map(convert_pdf_to_text, pdf_files), total=len(pdf_files)))

if __name__ == "__main__":
    main()
