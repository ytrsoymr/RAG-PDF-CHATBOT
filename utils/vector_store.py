from langchain_huggingface import HuggingFaceEmbeddings  # ✅ Updated import
from langchain_chroma import Chroma  # ✅ Updated import

from config import EMBEDDING_MODEL, CHROMA_DB_PATH

def store_embeddings(chunks):
    """Generate embeddings and store them in ChromaDB."""
    
    # Load the embedding model
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Store the embeddings in ChromaDB
    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DB_PATH)
    
    print("✅ Embeddings stored successfully in ChromaDB!")
    return db

def load_vector_store():
    """Load the stored ChromaDB vector store."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    return db

