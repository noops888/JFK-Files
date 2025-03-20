# Step-by-Step Implementation Guide for JFK Files Processing

This guide provides detailed instructions for implementing both the fast path extraction pipeline and the thorough secondary processing pipeline for the JFK Files collection. The instructions are designed to be accessible for non-technical users while providing all necessary details for successful implementation.

## Prerequisites

Before starting, ensure you have:

- A computer with at least 16GB RAM and 100GB free storage
- Python 3.8+ installed (instructions included below)
- Basic familiarity with running commands in a terminal/command prompt
- Your JFK Files collection organized in a single directory or structure

## Part 1: Setting Up Your Environment

### Step 1: Install Python (if not already installed)

**For Windows:**
1. Download the Python installer from [python.org](https://www.python.org/downloads/)
2. Run the installer, checking "Add Python to PATH"
3. Click "Install Now"
4. Verify installation by opening Command Prompt and typing: `python --version`

**For macOS:**
1. Install Homebrew if not already installed: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
2. Install Python: `brew install python`
3. Verify installation: `python3 --version`

**For Linux (Ubuntu/Debian):**
1. Update package lists: `sudo apt update`
2. Install Python: `sudo apt install python3 python3-pip`
3. Verify installation: `python3 --version`

### Step 2: Create a Project Directory

1. Create a new folder for the project:
   - Windows: `mkdir JFK_Files_Project`
   - macOS/Linux: `mkdir JFK_Files_Project`

2. Create subdirectories:
```bash
cd JFK_Files_Project
mkdir original_files
mkdir extracted_text
mkdir cleaned_text
mkdir problematic_files
mkdir database
```

3. Copy your JFK Files to the `original_files` directory (or create a symbolic link to save space)

### Step 3: Install Required Libraries

Create a file named `requirements.txt` with the following content:

```
pypdf2==3.0.1
pdfplumber==0.10.3
langchain==0.1.11
langchain_community==0.0.29
sentence-transformers==2.5.1
chromadb==0.4.22
unstructured==0.12.4
tqdm==4.66.2
numpy==1.26.4
regex==2023.12.25
```

Install the requirements:
```bash
pip install -r requirements.txt
```

For the secondary pipeline, you'll also need:
```bash
pip install ocrmypdf pytesseract pillow opencv-python-headless
```

## Part 2: Fast Path Extraction Implementation

### Step 1: Create Configuration File

Create a file named `config.py` with the following content:

```python
# Configuration settings for JFK Files processing

# File paths
ORIGINAL_FILES_DIR = "original_files"
EXTRACTED_TEXT_DIR = "extracted_text"
CLEANED_TEXT_DIR = "cleaned_text"
PROBLEMATIC_FILES_DIR = "problematic_files"
DATABASE_DIR = "database"

# Processing settings
MIN_TEXT_LENGTH = 100  # Minimum characters to consider successful extraction
BATCH_SIZE = 500  # Number of files to process in each batch

# Chunking settings
CHUNK_SIZE = 500  # tokens
CHUNK_OVERLAP = 100  # tokens

# Embedding settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
```

### Step 2: Create Fast Path Extraction Script

Create a file named `fast_extract.py`:

```python
import os
import sys
import pypdf2
import pdfplumber
from pathlib import Path
import config
import re
import shutil
from tqdm import tqdm

def extract_text_with_pypdf2(pdf_path):
    """Extract text using PyPDF2"""
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = pypdf2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n\n"
        return text
    except Exception as e:
        print(f"PyPDF2 error with {pdf_path}: {str(e)}")
        return ""

def extract_text_with_pdfplumber(pdf_path):
    """Extract text using pdfplumber"""
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n\n"
        return text
    except Exception as e:
        print(f"pdfplumber error with {pdf_path}: {str(e)}")
        return ""

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

def process_pdf_file(pdf_path, output_dir, problematic_dir):
    """Process a single PDF file"""
    # Create output filename
    rel_path = pdf_path.relative_to(Path(config.ORIGINAL_FILES_DIR))
    output_path = Path(output_dir) / rel_path.with_suffix('.txt')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path.parent, exist_ok=True)
    
    # Extract text with PyPDF2 first
    text = extract_text_with_pypdf2(pdf_path)
    
    # If not enough text, try pdfplumber
    if len(text.strip()) < config.MIN_TEXT_LENGTH:
        text = extract_text_with_pdfplumber(pdf_path)
    
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Check if we got enough text
    if len(cleaned_text.strip()) < config.MIN_TEXT_LENGTH:
        # Mark as problematic
        prob_path = Path(problematic_dir) / rel_path
        os.makedirs(prob_path.parent, exist_ok=True)
        shutil.copy2(pdf_path, prob_path)
        return False
    
    # Write the text to the output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
    
    return True

def main():
    # Get all PDF files in the original directory
    original_dir = Path(config.ORIGINAL_FILES_DIR)
    extracted_dir = Path(config.EXTRACTED_TEXT_DIR)
    problematic_dir = Path(config.PROBLEMATIC_FILES_DIR)
    
    # Create directories if they don't exist
    os.makedirs(extracted_dir, exist_ok=True)
    os.makedirs(problematic_dir, exist_ok=True)
    
    # Find all PDF files
    pdf_files = list(original_dir.glob("**/*.pdf"))
    total_files = len(pdf_files)
    print(f"Found {total_files} PDF files to process")
    
    # Process files with progress bar
    successful = 0
    problematic = 0
    
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        result = process_pdf_file(pdf_file, extracted_dir, problematic_dir)
        if result:
            successful += 1
        else:
            problematic += 1
    
    print(f"Processing complete!")
    print(f"Successfully processed: {successful} files")
    print(f"Problematic files: {problematic} files")
    print(f"Success rate: {successful/total_files*100:.2f}%")

if __name__ == "__main__":
    main()
```

### Step 3: Create Chunking and Embedding Script

Create a file named `chunk_and_embed.py`:

```python
import os
from pathlib import Path
import config
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm

def load_text_files(directory):
    """Load all text files from a directory"""
    text_files = []
    for file_path in Path(directory).glob("**/*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create metadata with file information
            metadata = {
                "source": str(file_path),
                "filename": file_path.name,
                "path": str(file_path.relative_to(directory))
            }
            
            text_files.append({"content": content, "metadata": metadata})
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    return text_files

def chunk_documents(documents):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    for doc in tqdm(documents, desc="Chunking documents"):
        doc_chunks = text_splitter.create_documents(
            [doc["content"]], 
            metadatas=[doc["metadata"]]
        )
        chunks.extend(doc_chunks)
    
    return chunks

def create_vector_store(chunks):
    """Create a vector store from document chunks"""
    # Initialize the embedding model
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    
    # Create the vector store
    print("Creating vector store...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=config.DATABASE_DIR
    )
    
    # Persist the vector store
    vector_store.persist()
    
    return vector_store

def main():
    # Load text files
    print("Loading text files...")
    documents = load_text_files(config.EXTRACTED_TEXT_DIR)
    print(f"Loaded {len(documents)} documents")
    
    # Chunk documents
    print("Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    # Create vector store
    print("Creating vector store...")
    vector_store = create_vector_store(chunks)
    print(f"Vector store created and saved to {config.DATABASE_DIR}")

if __name__ == "__main__":
    main()
```

### Step 4: Create Simple Query Interface

Create a file named `query_jfk_files.py`:

```python
import config
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import sys

def load_vector_store():
    """Load the vector store from disk"""
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    vector_store = Chroma(
        persist_directory=config.DATABASE_DIR,
        embedding_function=embeddings
    )
    return vector_store

def search_documents(query, vector_store, top_k=5):
    """Search for documents relevant to the query"""
    results = vector_store.similarity_search_with_score(query, k=top_k)
    return results

def display_results(results):
    """Display search results"""
    print("\nSearch Results:")
    print("===============")
    
    for i, (doc, score) in enumerate(results):
        print(f"\nResult {i+1} (Relevance: {score:.4f}):")
        print(f"Source: {doc.metadata['filename']}")
        print(f"Content: {doc.page_content[:300]}...")
        print("-" * 50)

def main():
    print("Loading vector store...")
    vector_store = load_vector_store()
    print("Vector store loaded!")
    
    print("\nJFK Files Query System")
    print("======================")
    print("Type 'exit' to quit\n")
    
    while True:
        query = input("Enter your question: ")
        if query.lower() == 'exit':
            break
        
        print("\nSearching for relevant documents...")
        results = search_documents(query, vector_store)
        
        display_results(results)
        
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()
```

### Step 5: Run the Fast Path Pipeline

Execute the scripts in the following order:

1. Extract text from PDFs:
```bash
python fast_extract.py
```

2. Create chunks and embeddings:
```bash
python chunk_and_embed.py
```

3. Query the knowledge base:
```bash
python query_jfk_files.py
```

## Part 3: Secondary Processing Implementation

### Step 1: Install Additional Dependencies

For the secondary processing pipeline, install additional dependencies:

**For Windows:**
1. Download and install Tesseract OCR from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
2. Add Tesseract to your PATH environment variable

**For macOS:**
```bash
brew install tesseract
```

**For Linux:**
```bash
sudo apt-get install tesseract-ocr
```

### Step 2: Create Secondary Processing Script

Create a file named `secondary_process.py`:

```python
import os
import sys
import subprocess
from pathlib import Path
import config
import shutil
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image
import pytesseract
import re

def preprocess_image(image_path):
    """Preprocess image for better OCR results"""
    # Read image
    img = cv2.imread(str(image_path))
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply deskew if needed
    # This is a simplified version - more advanced deskew might be needed
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    # Rotate the image to deskew it if angle is significant
    if abs(angle) > 0.5:
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    else:
        rotated = gray
    
    # Apply threshold to get black and white image
    _, binary = cv2.threshold(rotated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary

def ocr_with_tesseract(image):
    """Perform OCR using Tesseract"""
    # Convert OpenCV image to PIL Image
    pil_img = Image.fromarray(image)
    
    # Perform OCR
    text = pytesseract.image_to_string(pil_img, config='--psm 1')
    
    return text

def process_problematic_pdf(pdf_path, output_dir):
    """Process a problematic PDF file with enhanced OCR"""
    # Create output filename
    rel_path = pdf_path.relative_to(Path(config.PROBLEMATIC_FILES_DIR))
    output_path = Path(output_dir) / rel_path.with_suffix('.txt')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path.parent, exist_ok=True)
    
    try:
        # Create a temporary directory for images
        temp_dir = Path("temp_images")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Convert PDF to images
        cmd = ["pdftoppm", "-png", str(pdf_path), str(temp_dir / "page")]
        subprocess.run(cmd, check=True)
        
        # Process each image
        full_text = ""
        for img_path in sorted(temp_dir.glob("*.png")):
            # Preprocess image
            processed_img = preprocess_image(img_path)
            
            # Perform OCR
            page_text = ocr_with_tesseract(processed_img)
            
            # Add to full text
            full_text += page_text + "\n\n"
        
        # Clean the text
        cleaned_text = clean_text(full_text)
        
        # Write the text to the output file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        
        # Clean up temporary files
        shutil.rmtree(temp_dir)
        
        return True
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        return False

def clean_text(text):
    """Advanced cleaning of extracted text"""
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common OCR errors
    text = re.sub(r'([a-z])\s+([a-z])', r'\1\2', text)  # Fix split words
    text = re.sub(r'l\s*\'\s*([a-z])', r'I \1', text)   # Fix common 'I' misrecognition
    
    # Remove non-printable characters
    text = re.sub(r'[^\x20-\x7E\n]', '', text)
    
    # Normalize line breaks
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def main():
    # Get all PDF files in the problematic directory
    problematic_dir = Path(config.PROBLEMATIC_FILES_DIR)
    output_dir = Path(config.CLEANED_TEXT_DIR)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all PDF files
    pdf_files = list(problematic_dir.glob("**/*.pdf"))
    total_files = len(pdf_files)
    print(f"Found {total_files} problematic PDF files to process")
    
    # Process files with progress bar
    successful = 0
    failed = 0
    
    for pdf_file in tqdm(pdf_files, desc="Processing problematic PDFs"):
        result = process_problematic_pdf(pdf_file, output_dir)
        if result:
            successful += 1
        else:
            failed += 1
    
    print(f"Secondary processing complete!")
    print(f"Successfully processed: {successful} files")
    print(f"Failed: {failed} files")
    print(f"Success rate: {successful/total_files*100:.2f}%")

if __name__ == "__main__":
    main()
```

### Step 3: Create Integration Script

Create a file named `integrate_results.py`:

```python
import os
from pathlib import Path
import config
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm

def load_text_files(directory):
    """Load all text files from a directory"""
    text_files = []
    for file_path in Path(directory).glob("**/*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create metadata with file information
            metadata = {
                "source": str(file_path),
                "filename": file_path.name,
                "path": str(file_path.relative_to(directory)),
                "processing": "secondary"
            }
            
            text_files.append({"content": content, "metadata": metadata})
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    return text_files

def chunk_documents(documents):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    for doc in tqdm(documents, desc="Chunking documents"):
        doc_chunks = text_splitter.create_documents(
            [doc["content"]], 
            metadatas=[doc["metadata"]]
        )
        chunks.extend(doc_chunks)
    
    return chunks

def update_vector_store(chunks):
    """Update the vector store with new document chunks"""
    # Initialize the embedding model
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    
    # Load existing vector store
    print("Loading existing vector store...")
    vector_store = Chroma(
        persist_directory=config.DATABASE_DIR,
        embedding_function=embeddings
    )
    
    # Add new documents
    print("Adding new documents to vector store...")
    vector_store.add_documents(chunks)
    
    # Persist the vector store
    vector_store.persist()
    
    return vector_store

def main():
    # Load text files from secondary processing
    print("Loading secondary processed text files...")
    documents = load_text_files(config.CLEANED_TEXT_DIR)
    print(f"Loaded {len(documents)} documents from secondary processing")
    
    # Chunk documents
    print("Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    # Update vector store
    print("Updating vector store...")
    vector_store = update_vector_store(chunks)
    print(f"Vector store updated with secondary processed documents")

if __name__ == "__main__":
    main()
```

### Step 4: Run the Secondary Processing Pipeline

Execute the scripts in the following order:

1. Process problematic PDFs:
```bash
python secondary_process.py
```

2. Integrate results with the main knowledge base:
```bash
python integrate_results.py
```

3. Query the updated knowledge base:
```bash
python query_jfk_files.py
```

## Part 4: Practical Implementation Strategy

### Recommended Workflow

1. **Start Small**: Begin with a subset of your collection (1,000-5,000 files)
2. **Run Fast Path First**: Process the subset with the fast path pipeline
3. **Evaluate Results**: Check extraction quality and success rate
4. **Process Problematic Files**: Apply secondary processing to failed files
5. **Test Querying**: Try various queries to test the knowledge base
6. **Scale Up**: Gradually process larger batches of files
7. **Monitor and Adjust**: Refine parameters based on results

### Processing in Batches

If your collection is too large to process at once:

1. Create subdirectories in `original_files` (e.g., batch1, batch2)
2. Move manageable batches of files into each subdirectory
3. Update the `ORIGINAL_FILES_DIR` in config.py for each batch
4. Run the pipeline on each batch
5. The vector store will accumulate results from all batches

### Handling Memory Limitations

If you encounter memory issues:

1. Reduce batch sizes in the scripts
2. Process fewer files at a time
3. Close other applications while processing
4. Consider upgrading RAM if possible
5. Use a cloud computing service for processing (e.g., Google Colab)

## Part 5: Troubleshooting Common Issues

### Text Extraction Issues

- **Problem**: No text extracted from PDF
  - **Solution**: Check if PDF is password-protected or has security settings
  
- **Problem**: Garbled text in extraction
  - **Solution**: Try secondary processing with enhanced OCR

### OCR Processing Issues

- **Problem**: OCR process crashes
  - **Solution**: Reduce batch size, process one file at a time
  
- **Problem**: Poor OCR quality
  - **Solution**: Adjust preprocessing parameters, try different OCR engines

### Embedding Issues

- **Problem**: Out of memory during embedding
  - **Solution**: Reduce batch size, process fewer documents at once
  
- **Problem**: Slow embedding process
  - **Solution**: Use a smaller embedding model, process in smaller batches

### Query Issues

- **Problem**: Irrelevant search results
  - **Solution**: Adjust chunk size, try different embedding models
  
- **Problem**: Missing information in results
  - **Solution**: Check extraction quality, ensure all files are processed

## Conclusion

This implementation guide provides a comprehensive approach to processing your JFK Files collection. By following the fast path extraction first and then applying secondary processing to problematic files, you can efficiently build a queryable knowledge base with minimal technical expertise.

Remember to start small, validate your results, and scale up gradually. This approach allows you to make progress quickly while ensuring the quality of your knowledge base.

If you encounter persistent issues or need to process particularly challenging documents, consider seeking assistance from technical communities or exploring commercial OCR services for those specific files.
