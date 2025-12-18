import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

DB_PATH = "vectorstore/faiss_index"


def load_components():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0
    )

    return retriever, llm


def run_qa(question: str) -> str:
    retriever, llm = load_components()

    # ✅ NEW LangChain way
    docs = retriever.invoke(question)

    if not docs:
        return "I don’t know. The documents do not contain relevant information."

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are a helpful assistant.
Answer ONLY using the context below.
If the answer is not in the context, say "I don’t know".

Context:
{context}

Question:
{question}
"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content

    except Exception as e:
        print("⚠️ LLM unavailable:", str(e))
        return "I don’t know. The language model is currently unavailable."


if __name__ == "__main__":
    question = "What problem does retrieval-augmented generation solve?"
    answer = run_qa(question)

    print("\nQuestion:")
    print(question)
    print("\nAnswer:")
    print(answer)
