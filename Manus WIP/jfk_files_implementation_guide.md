# Step-by-Step Implementation Guide for JFK Files Processing

This guide provides detailed instructions for implementing the budget-friendly processing pipeline for the JFK Files collection. Each step is designed to be accessible for technical novices while maximizing efficiency and minimizing costs.

## Prerequisites

Before starting, ensure you have:

- A computer with at least 16GB RAM and 100GB free storage
- Basic familiarity with command line operations
- Python 3.8+ installed
- Git installed
- Internet connection for downloading tools and libraries

## Stage 1: Setting Up the Environment

### Step 1: Create a Project Directory

```bash
# Create a project directory
mkdir jfk_files_project
cd jfk_files_project

# Create subdirectories for each processing stage
mkdir -p original_files ocr_processed extracted_text chunks database
```

### Step 2: Install Required Tools

```bash
# Install Python dependencies
pip install ocrmypdf pypdf2 pdfplumber langchain langchain_community sentence-transformers chromadb llama-index unstructured

# Install system dependencies for OCRmyPDF
sudo apt-get update
sudo apt-get install -y ocrmypdf tesseract-ocr tesseract-ocr-eng parallel
```

### Step 3: Set Up Configuration File

Create a file named `config.py` with the following content:

```python
# Configuration settings for JFK Files processing

# File paths
ORIGINAL_FILES_DIR = "original_files"
OCR_PROCESSED_DIR = "ocr_processed"
EXTRACTED_TEXT_DIR = "extracted_text"
CHUNKS_DIR = "chunks"
DATABASE_DIR = "database"

# OCR settings
OCR_BATCH_SIZE = 100
OCR_PARALLEL_JOBS = 2  # Adjust based on your CPU cores

# Text extraction settings
EXTRACTION_BATCH_SIZE = 500

# Chunking settings
CHUNK_SIZE = 500  # tokens
CHUNK_OVERLAP = 100  # tokens

# Embedding settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 1000

# Database settings
VECTOR_DB_TYPE = "chroma"
```

## Stage 2: OCR Processing

### Step 1: Organize Original Files

```bash
# Copy or move your JFK Files to the original_files directory
# Example (adjust paths as needed):
# cp -r /path/to/jfk_files/* original_files/
```

### Step 2: Create OCR Processing Script

Create a file named `ocr_processor.py`:

```python
import os
import subprocess
from pathlib import Path
import config

def create_batch_list(file_list, batch_size):
    """Split a list into batches of specified size"""
    for i in range(0, len(file_list), batch_size):
        yield file_list[i:i + batch_size]

def process_pdf_with_ocr(input_file, output_file):
    """Process a single PDF file with OCRmyPDF"""
    try:
        # Skip if output file already exists
        if os.path.exists(output_file):
            print(f"Skipping {input_file} - already processed")
            return True
            
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Run OCRmyPDF
        cmd = ["ocrmypdf", "--skip-text", input_file, output_file]
        subprocess.run(cmd, check=True)
        print(f"Successfully processed {input_file}")
        return True
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")
        return False

def main():
    # Get all PDF files in the original directory
    original_dir = Path(config.ORIGINAL_FILES_DIR)
    output_dir = Path(config.OCR_PROCESSED_DIR)
    
    pdf_files = list(original_dir.glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Process files in batches
    for batch_num, batch in enumerate(create_batch_list(pdf_files, config.OCR_BATCH_SIZE)):
        print(f"Processing batch {batch_num+1}")
        
        # Create a file with paths for GNU Parallel
        batch_file = f"batch_{batch_num}.txt"
        with open(batch_file, "w") as f:
            for pdf_file in batch:
                # Maintain the same directory structure in the output
                rel_path = pdf_file.relative_to(original_dir)
                output_path = output_dir / rel_path
                f.write(f"{pdf_file}\t{output_path}\n")
        
        # Run GNU Parallel
        cmd = [
            "parallel", 
            "--colsep", "\t", 
            "-j", str(config.OCR_PARALLEL_JOBS),
            "ocrmypdf --skip-text {1} {2}"
        ]
        
        try:
            subprocess.run(cmd + [f":::", batch_file], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error in batch {batch_num+1}: {str(e)}")
        
        # Clean up batch file
        os.remove(batch_file)

if __name__ == "__main__":
    main()
```

### Step 3: Run OCR Processing

```bash
# Run the OCR processing script
python ocr_processor.py
```

## Stage 3: Text Extraction

### Step 1: Create Text Extraction Script

Create a file named `text_extractor.py`:

```python
import os
from pathlib import Path
import config
import pypdf2
import pdfplumber
from unstructured.partition.pdf import partition_pdf

def extract_text_from_pdf(pdf_path, output_path):
    """Extract text from a PDF file using multiple methods"""
    try:
        # First try with PyPDF2
        text = extract_with_pypdf2(pdf_path)
        
        # If PyPDF2 didn't get much text, try pdfplumber
        if len(text.strip()) < 100:
            text = extract_with_pdfplumber(pdf_path)
        
        # If still not much text, try unstructured
        if len(text.strip()) < 100:
            text = extract_with_unstructured(pdf_path)
        
        # Write the extracted text to the output file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"Successfully extracted text from {pdf_path}")
        return True
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {str(e)}")
        return False

def extract_with_pypdf2(pdf_path):
    """Extract text using PyPDF2"""
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = pypdf2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n\n"
    return text

def extract_with_pdfplumber(pdf_path):
    """Extract text using pdfplumber"""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or "" + "\n\n"
    return text

def extract_with_unstructured(pdf_path):
    """Extract text using unstructured"""
    elements = partition_pdf(pdf_path)
    return "\n\n".join([str(element) for element in elements])

def main():
    # Get all PDF files in the OCR processed directory
    ocr_dir = Path(config.OCR_PROCESSED_DIR)
    output_dir = Path(config.EXTRACTED_TEXT_DIR)
    
    pdf_files = list(ocr_dir.glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} PDF files to extract text from")
    
    # Process files in batches
    for i, pdf_file in enumerate(pdf_files):
        if i % 100 == 0:
            print(f"Processing file {i+1}/{len(pdf_files)}")
        
        # Maintain the same directory structure in the output
        rel_path = pdf_file.relative_to(ocr_dir)
        output_path = output_dir / rel_path.with_suffix('.txt')
        
        extract_text_from_pdf(pdf_file, output_path)

if __name__ == "__main__":
    main()
```

### Step 2: Run Text Extraction

```bash
# Run the text extraction script
python text_extractor.py
```

## Stage 4: Chunking & Embedding

### Step 1: Create Chunking and Embedding Script

Create a file named `chunk_and_embed.py`:

```python
import os
from pathlib import Path
import config
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

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
    for doc in documents:
        doc_chunks = text_splitter.create_documents(
            [doc["content"]], 
            metadatas=[doc["metadata"]]
        )
        chunks.extend(doc_chunks)
    
    return chunks

def create_vector_store(chunks):
    """Create a vector store from document chunks"""
    # Initialize the embedding model
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    
    # Create the vector store
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

### Step 2: Run Chunking and Embedding

```bash
# Run the chunking and embedding script
python chunk_and_embed.py
```

## Stage 5: RAG Implementation

### Step 1: Create a Simple Query Interface

Create a file named `query_jfk_files.py`:

```python
import config
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

def load_vector_store():
    """Load the vector store from disk"""
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    vector_store = Chroma(
        persist_directory=config.DATABASE_DIR,
        embedding_function=embeddings
    )
    return vector_store

def setup_retrieval_qa(vector_store):
    """Set up the retrieval QA chain"""
    # Initialize the LLM
    llm = Ollama(model="llama2")
    
    # Create the retrieval QA chain
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    return qa_chain

def main():
    print("Loading vector store...")
    vector_store = load_vector_store()
    
    print("Setting up retrieval QA chain...")
    qa_chain = setup_retrieval_qa(vector_store)
    
    print("\nJFK Files Query System")
    print("======================")
    print("Type 'exit' to quit\n")
    
    while True:
        query = input("Enter your question: ")
        if query.lower() == 'exit':
            break
        
        print("\nSearching for answer...")
        result = qa_chain({"query": query})
        
        print("\nAnswer:")
        print(result["result"])
        
        print("\nSources:")
        for i, doc in enumerate(result["source_documents"]):
            print(f"{i+1}. {doc.metadata['filename']}")
        
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()
```

### Step 2: Install Ollama (Local LLM)

```bash
# Install Ollama for local LLM inference
curl -fsSL https://ollama.com/install.sh | sh

# Pull the llama2 model
ollama pull llama2
```

### Step 3: Run the Query Interface

```bash
# Run the query interface
python query_jfk_files.py
```

## Processing in Smaller Batches

If your computer has limited resources, you can process the files in smaller batches:

1. Divide your JFK Files into smaller subsets (e.g., by year or release)
2. Process each subset separately by updating the config.py file
3. Merge the vector stores after processing all subsets

## Troubleshooting Common Issues

### OCR Processing Issues

- **Error**: "OCR failed for file X"
  - **Solution**: Try processing the file individually with different OCR settings
  
- **Error**: "Out of memory"
  - **Solution**: Reduce the OCR_PARALLEL_JOBS in config.py

### Text Extraction Issues

- **Error**: "Failed to extract text from file X"
  - **Solution**: Try a different extraction method or manually review the file

### Embedding Issues

- **Error**: "CUDA out of memory"
  - **Solution**: Reduce EMBEDDING_BATCH_SIZE in config.py
  
- **Error**: "Model not found"
  - **Solution**: Check internet connection or use a different embedding model

## Next Steps and Improvements

Once you have the basic pipeline working, consider these improvements:

1. **Quality Assessment**: Randomly sample processed files to check quality
2. **Custom Preprocessing**: Add specific preprocessing for JFK Files format
3. **Web Interface**: Create a simple web interface using Streamlit or Gradio
4. **Metadata Extraction**: Extract and use document metadata for better retrieval
5. **Advanced Filtering**: Implement filtering by date, document type, or source

## Conclusion

This implementation guide provides a step-by-step approach to processing the JFK Files collection. The modular design allows for flexibility and incremental processing, making it accessible for technical novices while still being powerful enough to handle the large collection efficiently.

Remember to back up your original files before processing and to monitor system resources during processing. If you encounter persistent issues, consider processing smaller batches or adjusting the configuration parameters.
