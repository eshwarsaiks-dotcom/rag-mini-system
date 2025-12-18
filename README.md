# Mini RAG System – Technical Challenge Submission

## Overview
This repository contains a minimal yet complete Retrieval-Augmented Generation (RAG) system built as part of a Data/ML Engineer technical challenge.

The system ingests a small corpus of PDF documents, chunks and embeds them, stores embeddings in a local FAISS vector index, and retrieves relevant context at query time. Retrieved chunks can optionally be passed to an LLM for answer generation.

The implementation intentionally prioritizes clarity, simplicity, and extensibility over heavy abstraction.

---

## Architecture & Design Choices

- **Chunking**
  - RecursiveCharacterTextSplitter  
  - Chunk size: 800 characters  
  - Overlap: 150 characters  
  - Chosen to balance semantic coherence with retrieval recall.

- **Embeddings**
  - Sentence-Transformers (local inference)
  - Cost-free, fast, and sufficient for small to medium corpora.

- **Vector Store**
  - FAISS (local index)
  - Simple, performant, and easy to swap for managed vector databases later.

- **Retrieval**
  - Top-k similarity search
  - Designed to be easily extended to hybrid or reranking strategies.

- **Generation**
  - Optional OpenAI chat model
  - Gracefully degrades when API quota or credentials are unavailable.

---

## Running the System

1. Ingest and embed documents:
   ```bash
   python src/ingest.py
