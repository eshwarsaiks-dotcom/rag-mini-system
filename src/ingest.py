import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

PDF_DIR = "data/pdfs"
VECTOR_DB_PATH = "vectorstore/faiss_index"


def load_pdfs():
    documents = []
    for file in os.listdir(PDF_DIR):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(PDF_DIR, file))
            docs = loader.load()
            documents.extend(docs)
    return documents


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return splitter.split_documents(documents)


def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs("vectorstore", exist_ok=True)
    vectorstore.save_local(VECTOR_DB_PATH)

    return vectorstore


if __name__ == "__main__":
    docs = load_pdfs()
    chunks = chunk_documents(docs)

    print(f"Loaded documents: {len(docs)}")
    print(f"Total chunks created: {len(chunks)}")

    create_vector_store(chunks)

    print("FAISS vector store created and saved ✅")
