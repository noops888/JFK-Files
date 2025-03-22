import os
import warnings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import argparse

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Constants
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_STORE_PATH = "database"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def load_vector_store():
    """Load the vector store from disk"""
    print("Loading vector store...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Check if the vector store directory exists
    if not os.path.exists(VECTOR_STORE_PATH):
        raise ValueError(f"Vector store directory '{VECTOR_STORE_PATH}' not found!")
    
    vector_store = Chroma(persist_directory=VECTOR_STORE_PATH, embedding_function=embeddings)
    
    # Get collection info
    collection = vector_store.get()
    print(f"Loaded {len(collection['ids'])} documents from vector store")
    
    return vector_store

def create_qa_chain(vector_store, model_type="claude"):
    """Create a question-answering chain with the specified model"""
    if model_type == "claude":
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        llm = ChatAnthropic(
            model="claude-3-opus-20240229",
            anthropic_api_key=ANTHROPIC_API_KEY,
            temperature=0.3,  # Lower temperature for more consistent answers
            max_tokens=4000  # Increased for longer responses
        )
    elif model_type == "gemini":
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3,  # Lower temperature for more consistent answers
            max_output_tokens=4000  # Increased for longer responses
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Custom prompt template with more explicit instructions
    prompt_template = """You are a helpful assistant that answers questions about the JFK assassination based on the provided context. 
    Your task is to carefully analyze the provided context and answer the question based ONLY on the information given in the context.
    
    Important rules:
    1. ONLY use information from the provided context
    2. If the context contains information about the topic, use it
    3. If the context doesn't contain relevant information, say "Based on the provided context, I don't have enough information to answer this question"
    4. Do not make assumptions or add information not present in the context
    5. If the context is incomplete or unclear, acknowledge this in your answer
    6. When possible, provide specific details, dates, and names from the context
    7. If there are multiple relevant pieces of information, combine them into a comprehensive answer

    Context:
    {context}

    Question: {question}

    Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Create the chain with improved retrieval parameters
    chain_type_kwargs = {"prompt": PROMPT}
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_type="similarity_score_threshold",  # Use threshold-based search
            search_kwargs={
                "k": 20,  # Increased number of documents
                "score_threshold": 0.3,  # Lowered threshold to get more context
                "fetch_k": 50  # Increased number of candidates to consider
            }
        ),
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True
    )
    return qa_chain

def main():
    parser = argparse.ArgumentParser(description='Query JFK Files using LLM')
    parser.add_argument('--model', choices=['claude', 'gemini'], default='claude',
                      help='Choose the LLM model to use (claude or gemini)')
    args = parser.parse_args()

    print("\nJFK Files Query System")
    print("======================")
    print("Type 'exit' to quit\n")

    try:
        # Load vector store and create QA chain
        vector_store = load_vector_store()
        qa_chain = create_qa_chain(vector_store, args.model)

        while True:
            query = input("\nEnter your question: ").strip()
            if query.lower() == 'exit':
                break

            print("\nSearching for answer...")
            try:
                # First, let's see what documents are being retrieved
                docs = vector_store.similarity_search_with_score(query, k=20)
                print(f"\nFound {len(docs)} relevant documents")
                print("Top document scores:")
                for doc, score in docs[:5]:  # Show top 5 documents
                    print(f"- {doc.metadata['source']} (score: {score:.3f})")
                
                # Use invoke instead of __call__
                result = qa_chain.invoke({"query": query})
                print("\nAnswer:", result["result"])
                print("\nSources:")
                for doc in result["source_documents"]:
                    print(f"- {doc.metadata['source']}")
            except Exception as e:
                print(f"\nError: {str(e)}")
    except Exception as e:
        print(f"\nFatal error: {str(e)}")

if __name__ == "__main__":
    main() 