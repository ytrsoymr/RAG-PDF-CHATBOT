import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_pdfs(pdf_dir):
    """Load all PDFs in a directory, split them into chunks, and return them."""
    
    # Load all PDFs from the directory
    loader = PyPDFDirectoryLoader(pdf_dir)
    documents = loader.load()

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_chunks = text_splitter.split_documents(documents)
    
    return all_chunks
