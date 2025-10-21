from langchain_chroma import Chroma
from langchain_classic import hub
from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever,
)
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from bluebridge.config import Config

RAG_PROMPT = hub.pull("rlm/rag-prompt")


def build_rag_chain(vector_store: Chroma, config: Config):
    """
    Compose a simple RAG chain: retriever -> prompt -> LLM -> text.
    """

    base = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": config.mmr_k,
            "fetch_k": config.fetch_k,
            "lambda_mult": config.mmr_lambda,
        },
    )

    retriever = ContextualCompressionRetriever(
        base_retriever=base,
        base_compressor=config.reranker_model.instance(config.top_k),
    )

    # Format retrieved docs into a numbered context block
    def format_docs(docs):
        lines = []
        for i, d in enumerate(docs, 1):
            src = d.metadata.get("source") or d.metadata.get("file_path") or "unknown"
            page = d.metadata.get("page")
            tag = f"{src}" + (f":p{page}" if page is not None else "")
            lines.append(f"[{i}] ({tag})\n{d.page_content}")
        return "\n\n---\n\n".join(lines)

    rag_chain = (
        {
            "context": retriever | (lambda docs: format_docs(docs)),
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | config.llm_model.instance()
        | StrOutputParser()
    )

    return rag_chain
