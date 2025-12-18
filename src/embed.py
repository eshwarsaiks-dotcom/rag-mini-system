from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from ingest import load_pdfs, chunk_documents
import os

VECTOR_DIR = "data/vector_store"

if __name__ == "__main__":
    print("Loading PDFs...")
    documents = load_pdfs()

    print("Chunking documents...")
    chunks = chunk_documents(documents)

    print("Creating embeddings...")
    embeddings = OpenAIEmbeddings()

    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(VECTOR_DIR, exist_ok=True)
    vectorstore.save_local(VECTOR_DIR)

    print("✅ Vector store created and saved!")
