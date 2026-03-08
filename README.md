# RAG Document Intelligence System

A Retrieval-Augmented Generation (RAG) pipeline that enables natural language Q&A over PDF documents. The system ingests PDFs, builds a semantic vector index, and uses Claude as the language model to generate accurate, context-grounded answers with source citations.

## How It Works

```
PDF Documents ──► Text Extraction ──► Chunking ──► Embedding ──► Vector Store
                                                                      │
User Question ──► Query Embedding ──► Cosine Similarity Search ◄──────┘
                                              │
                                     Top-K Chunks Retrieved
                                              │
                                    Claude API (with context) ──► Grounded Answer + Sources
```

1. **Ingestion** — PDFs are parsed with PyPDF2, split into overlapping 1000-character chunks (200-char overlap), and encoded into 384-dimensional vectors using the `all-MiniLM-L6-v2` sentence-transformer model. Embeddings and metadata are persisted to disk.

2. **Retrieval** — When a user asks a question, the query is encoded with the same embedding model. Cosine similarity is computed against all stored vectors to find the top 5 most relevant chunks.

3. **Generation** — Retrieved chunks are injected into a structured prompt and sent to the Claude API. Claude generates an answer grounded strictly in the provided context, preventing hallucination.

## Tech Stack

| Component | Technology |
|---|---|
| Language | **Python** |
| LLM | **Claude API** (claude-sonnet-4-6) |
| Embeddings | **sentence-transformers** (all-MiniLM-L6-v2) |
| Vector Math | **NumPy** (cosine similarity, compressed storage) |
| PDF Parsing | **PyPDF2** |

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rag-system.git
cd rag-system

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your Anthropic API key
export ANTHROPIC_API_KEY="your-api-key"
```

## Usage

### 1. Ingest Documents

Place PDF files in the `documents/` folder, then run:

```bash
python ingest.py
```

This extracts text from each PDF, chunks it, generates embeddings, and stores everything in `vector_db/`.

### 2. Query Your Documents

```bash
python query.py
```

Enter natural language questions at the interactive prompt. The system retrieves relevant context and returns a Claude-generated answer with source references and similarity scores.

```
Ask a question (or 'quit' to exit): What is retrieval-augmented generation?

Answer: ...

Sources:
  - test_ai_overview.pdf (chunk 1, distance: 0.42)
```

## Skills Demonstrated

- **LLM Integration** — End-to-end integration with the Claude API, including structured prompt construction and response handling
- **Vector Embeddings** — Semantic encoding of text using transformer models, with a custom NumPy-based vector store supporting cosine similarity search
- **Prompt Engineering** — Context-injection prompts that ground LLM responses in retrieved documents to minimize hallucination
- **AI Pipeline Architecture** — Modular design separating ingestion, storage, retrieval, and generation into composable components
