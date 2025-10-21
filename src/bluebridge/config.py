from pathlib import Path

from pydantic import BaseModel

from bluebridge.models import EmbeddingModel, LLMModel


class Config(BaseModel):
    docs_dir: Path = Path("./docs")
    vector_store_dir: Path = Path("./chroma_db")
    models_dir: Path = Path("./models")
    questions_file: Path = Path("./questions.json")

    rebuild_vector_store: bool = True

    chunk_size: int = 240
    chunk_overlap: int = 30
    fetch_k: int = 5
    mmr_k: int = 5
    mmr_lambda: float = 0.3
    top_k: int = 3
    rerank: bool = True

    embedding: EmbeddingModel = EmbeddingModel.BAAI_BGE

    llm_model: LLMModel = LLMModel.OLLAMA_GEMMA2_2B
