import shutil

from langchain_chroma import Chroma
from langchain_text_splitters import TokenTextSplitter

from bluebridge.config import Config
from bluebridge.files import load_documents


def build_or_load_vectorstore(
    config: Config,
) -> Chroma:
    """
    Create (or load) a Chroma vector DB persisted on disk.
    """

    embedding_function = config.embedding.embedding_function(config.models_dir)

    if config.vector_store_dir.exists() and not config.rebuild_vector_store:
        print(f"[info] Using existing Chroma at {config.vector_store_dir}")
        vs = Chroma(
            persist_directory=str(config.vector_store_dir),
            embedding_function=embedding_function,
        )
        return vs

    shutil.rmtree(config.vector_store_dir, ignore_errors=True)

    print("[info] Building Chroma store...")
    docs = load_documents(config.docs_dir)

    splitter = TokenTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    print(f"[info] Split into {len(chunks)} chunks")

    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=str(config.vector_store_dir),
    )
    print(f"[info] Persisted at {config.vector_store_dir} (docs={len(chunks)})")
    return vs
