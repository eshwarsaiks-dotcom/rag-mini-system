# Mini RAG System – Challenge Submission

## Overview
This project implements a minimal Retrieval-Augmented Generation (RAG) system.
PDF documents are ingested, chunked, embedded, and stored in a FAISS vector store.
At query time, relevant chunks are retrieved and optionally passed to an LLM for answer generation.

## Architecture
- Chunking: RecursiveCharacterTextSplitter (800 chars, 150 overlap)
- Embeddings: Sentence-Transformers (local, cost-free)
- Vector Store: FAISS (local)
- Retrieval: Top-k similarity search
- Generation: Optional OpenAI LLM (gracefully handled if unavailable)

## Notes
- Embedding and retrieval work fully offline.
- LLM usage is optional and may be limited by API quota.
- The system is designed to be simple, modular, and extensible.
