from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

DB_PATH = "vectorstore/faiss_index"

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore


def retrieve_chunks(query: str, k: int = 3):
    vectorstore = load_vectorstore()
    docs = vectorstore.similarity_search(query, k=k)
    return docs


if __name__ == "__main__":
    query = "What is chain-of-knowledge?"
    results = retrieve_chunks(query)

    print(f"Retrieved {len(results)} chunks\n")
    for i, doc in enumerate(results, 1):
        print(f"--- Chunk {i} ---")
        print(doc.page_content[:300])
        print()
