import os
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm

# Configuration
EXTRACTED_TEXT_DIR = "2025"
DATABASE_DIR = "database"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

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
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
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
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Create the vector store
    print("Creating vector store...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DATABASE_DIR
    )
    
    # Persist the vector store
    vector_store.persist()
    
    return vector_store

def main():
    # Load text files
    print("Loading text files...")
    documents = load_text_files(EXTRACTED_TEXT_DIR)
    print(f"Loaded {len(documents)} documents")
    
    # Chunk documents
    print("Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    # Create vector store
    print("Creating vector store...")
    vector_store = create_vector_store(chunks)
    print(f"Vector store created and saved to {DATABASE_DIR}")

if __name__ == "__main__":
    main()
