import config
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configuration
DATABASE_DIR = "database"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def load_vector_store():
    """Load the vector store from disk"""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma(
        persist_directory=DATABASE_DIR,
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
