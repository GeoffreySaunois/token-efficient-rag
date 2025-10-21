from enum import Enum
from pathlib import Path
from typing import Any, Sequence

from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import ConfigDict
from sentence_transformers import CrossEncoder


class EmbeddingModel(str, Enum):
    OPENAI_LARGE = "openai_large"
    BGE_SMALL = "bge_small"

    def instance(self, models_dir: Path):
        match self:
            case EmbeddingModel.OPENAI_LARGE:
                return OpenAIEmbeddings(model="text-embedding-3-large")
            case EmbeddingModel.BGE_SMALL:
                return HuggingFaceEmbeddings(
                    model_name=f"{models_dir}/bge-small-en-v1.5",
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )


class LLMModel(str, Enum):
    OPENAI_GPT5 = "gpt-5"
    GEMMA2_2B = "gemma2:2b"
    GEMMA3_1B_IT = "gemma3:1b"
    QWEN2_5 = "qwen2.5:1.5b-instruct"

    def instance(self):
        match self:
            case LLMModel.OPENAI_GPT5:
                return ChatOpenAI(model=self.value, temperature=0)
            case LLMModel.GEMMA2_2B:
                return ChatOllama(model=self.value, temperature=0)
            case LLMModel.GEMMA3_1B_IT:
                return ChatOllama(model=self.value, temperature=0)
            case LLMModel.QWEN2_5:
                return ChatOllama(model=self.value, temperature=0)


class RerankerModel(str, Enum):
    MS_MARCO = "ms-marco"
    BGE_LARGE = "bge-large"
    ZERANK_SMALL = "zerank-small"

    def instance(self, top_k):
        match self:
            case RerankerModel.MS_MARCO:
                return CrossEncoderReranker(
                    model=HuggingFaceCrossEncoder(
                        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
                    ),
                    top_n=top_k,
                )

            case RerankerModel.BGE_LARGE:
                return CrossEncoderReranker(
                    model=HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-large"),
                    top_n=top_k,
                )
            case RerankerModel.ZERANK_SMALL:
                return ZerankReranker(
                    model=CrossEncoder(
                        model_name="zeroentropy/zerank-1-small",
                        trust_remote_code=True,
                        device="cpu",
                    ),
                    top_n=top_k,
                )


class ZerankReranker(BaseDocumentCompressor):
    """
    Adapter to use ZeroRank CrossEncoder as a document compressor (and bypass LangChain's Pydantic checks).
    """

    model: Any
    top_n: int

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        pairs = [(query, (d.page_content or "")) for d in documents]
        scores = self.model.predict(pairs)  # type: ignore
        ranked = sorted(zip(documents, scores), key=lambda x: float(x[1]), reverse=True)
        return [doc for doc, _ in ranked[: self.top_n]]
