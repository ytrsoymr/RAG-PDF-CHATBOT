from utils.pdf_loader import load_and_split_pdfs

dir_path="./data"
chunks=load_and_split_pdfs(dir_path)

from utils.vector_store import store_embeddings,load_vector_store
store_embeddings(chunks)
db=load_vector_store()
from utils.query_engine import retrieve_similar_chunks
query="what is attention"
similar_chunks=retrieve_similar_chunks(query,k=3)
from utils.llm import generate_answer
ans=generate_answer(query=query,context=similar_chunks)
print(ans)