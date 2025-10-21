from langchain_chroma import Chroma
from langchain_classic import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap, RunnablePassthrough

from bluebridge.config import Config

RAG_PROMPT = hub.pull("rlm/rag-prompt")


def build_rag_chain(vector_store: Chroma, config: Config):
    """
    Compose a simple RAG chain: retriever -> prompt -> LLM -> text.
    """
    if config.rerank:
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": config.top_k,
                "fetch_k": config.fetch_k,
                "lambda_mult": 0.5,
            },
        )
    else:
        retriever = vector_store.as_retriever(search_kwargs={"k": config.top_k})

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
        | config.llm_model.llm_instance()
        | StrOutputParser()
    )

    return rag_chain
