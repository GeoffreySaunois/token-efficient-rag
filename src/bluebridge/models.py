from enum import Enum
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class EmbeddingModel(str, Enum):
    OPEN_AI = "openai"
    BAAI_BGE = "baai_bge"

    def embedding_function(self, models_dir: Path):
        match self:
            case EmbeddingModel.OPEN_AI:
                return OpenAIEmbeddings(model="text-embedding-3-large")
            case EmbeddingModel.BAAI_BGE:
                return HuggingFaceEmbeddings(
                    model_name=f"{models_dir}/bge-small-en-v1.5",
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )


class LLMModel(str, Enum):
    OPEN_AI_GPT5 = "gpt-5"
    OLLAMA_GEMMA2_2B = "gemma2:2b"

    def llm_instance(self):
        match self:
            case LLMModel.OPEN_AI_GPT5:
                return ChatOpenAI(model=self.value, temperature=0)
            case LLMModel.OLLAMA_GEMMA2_2B:
                return ChatOllama(model=self.value, temperature=0)
