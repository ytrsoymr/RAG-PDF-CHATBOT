import streamlit as st
import os
from utils.pdf_loader import load_and_split_pdfs
from utils.vector_store import store_embeddings, load_vector_store
from utils.query_engine import retrieve_similar_chunks
from utils.llm import generate_answer

# Streamlit UI
st.title("RAG Chatbot using ChromaDB and LangChain")
st.markdown("""
### Welcome to the RAG Chatbot! 🤖
This chatbot allows you to upload PDF documents, process their contents, and retrieve relevant information based on your queries. 
- Upload one or multiple PDFs.
- Choose chunk size for better context retrieval.
- Ask questions, and get AI-powered responses!
""")

uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# Chunk size selection
chunk_size = st.slider("Select chunk size for document processing:", min_value=200, max_value=2000, step=100, value=500)
chunk_overlap = st.slider("Select chunk overlap:", min_value=0, max_value=500, step=50, value=50)

temp_directory = "uploaded_pdfs"
os.makedirs(temp_directory, exist_ok=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        with open(os.path.join(temp_directory, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getvalue())
    
    st.success("PDFs uploaded successfully!")
    
    # Process PDFs
    with st.spinner("Processing documents..."):
        chunks = load_and_split_pdfs(temp_directory, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        store_embeddings(chunks)
        db = load_vector_store()
    st.success("Documents processed and stored in vector database!")

# User query input
query = st.text_input("Ask me anything:")

if query:
    with st.spinner("Retrieving relevant information..."):
        similar_chunks = retrieve_similar_chunks(query, k=3)
        ans = generate_answer(query=query, context=similar_chunks)
    
    st.subheader("Response:")
    st.write(ans)
