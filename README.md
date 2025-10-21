# 🚀 BlueBridge RAG System

A token-efficient Retrieval-Augmented Generation (RAG) system for context-aware question answering under strict token constraints.

## 🔧 Setup

### Prerequisites

- Python 3.13+
- [Ollama](https://ollama.ai/) installed and running (for local LLMs)

### Installation

1. **Install dependencies:**

   ```bash
   uv sync
   ```

2. **Download required models:**
   ```bash
   # Pull local LLMs via Ollama
   ollama pull gemma2:2b
   ollama pull gemma3:1b
   ollama pull qwen2.5:1.5b-instruct
   ```

## 🚀 Usage

### See CLI help

```bash
bluebridge --help
```

### Ask a Question

```bash
bluebridge ask --question "What happens if the QRC fails during boot?"
```

### Benchmark Mode

Run evaluation on all provided test questions:

```bash
bluebridge benchmark --llm "gemma3:1b" --embedding "bge_small" --rebuild
```

## 📊 Available Models

### LLM Models

- `gemma2:2b` - Default, good balance of speed/quality
- `gemma:1b` - Fastest, lowest resource usage
- `qwen2.5:1.5b-instruct` - Alternative small model
- `gpt-5` - OpenAI GPT-5 model (requires API key)

### Embedding Models

- `bge_small` - Default, local BGE model (384 dimensions)
- `openai_large` - OpenAI's text-embedding-3-large (requires API key)

### Reranker Models (configured but not yet integrated)

- `ms-marco` - Cross-encoder for relevance reranking
- `bge-large` - BAAI BGE reranker
- `zerank-small` - Zero-entropy reranker

## 🗂️ Project Structure

```
bluebridge/
├── docs/                    # Knowledge base (.md files)
├── src/bluebridge/          # Main package
│   ├── cli.py              # Command-line interface
│   ├── config.py           # Configuration management
│   ├── models.py           # Model definitions & loading
│   ├── rag.py              # RAG pipeline implementation
│   ├── vector_store.py     # Vector database management
│   └── files.py            # Document loading utilities
├── models/                  # Local model storage
├── chroma_db/              # Vector store persistence
├── questions.json          # Benchmark questions
└── pyproject.toml          # Project configuration
```
