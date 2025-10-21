import warnings
from pathlib import Path

from pydantic import BaseModel

from bluebridge.models import EmbeddingModel, LLMModel, RerankerModel

warnings.filterwarnings("ignore", message="Relevance scores must be between 0 and 1")


class Config(BaseModel):
    docs_dir: Path = Path("./docs")
    vector_store_dir: Path = Path("./chroma_db")
    models_dir: Path = Path("./models")
    questions_file: Path = Path("./questions.json")

    rebuild_vector_store: bool = True

    chunk_size: int = 240
    chunk_overlap: int = 30

    fetch_k: int = 5
    mmr_k: int = 3
    top_k: int = 3

    mmr_lambda: float = 0.3

    embedding_model: EmbeddingModel = EmbeddingModel.BGE_SMALL
    reranker_model: RerankerModel = RerankerModel.MS_MARCO
    llm_model: LLMModel = LLMModel.GEMMA3_1B_IT
