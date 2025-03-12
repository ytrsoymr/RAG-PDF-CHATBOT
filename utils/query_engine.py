from langchain_huggingface import HuggingFaceEmbeddings  # ✅ Updated import
from langchain_chroma import Chroma  # ✅ Updated import
from config import CHROMA_DB_PATH, EMBEDDING_MODEL

# Load embeddings model
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Load ChromaDB
db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)

def retrieve_similar_chunks(query: str, k=3):
    """Retrieve top-k most relevant document chunks from ChromaDB."""
    results = db.similarity_search(query, k=k)
    return [doc.page_content for doc in results]
