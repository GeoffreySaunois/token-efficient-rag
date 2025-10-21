from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader


def load_documents(docs_dir: Path):
    loader = DirectoryLoader(
        str(docs_dir),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={
            "encoding": "utf-8",
            "autodetect_encoding": True,
        },
    )
    docs = loader.load()

    if not docs:
        raise RuntimeError(f"No documents found under {docs_dir.resolve()}.")
    return docs
