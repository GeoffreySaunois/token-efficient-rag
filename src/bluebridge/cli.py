import json

from typer import Option, Typer

from bluebridge.config import Config
from bluebridge.models import EmbeddingModel, LLMModel
from bluebridge.rag import build_rag_chain
from bluebridge.vector_store import build_or_load_vectorstore

app = Typer(help="BlueBridge CLI")


@app.command(help="Ask a question to the BlueBridge knowledge base.")
def ask(
    question: str = Option(..., prompt="What is your question?"),
    rebuild: bool = Option(True, help="Whether to rebuild the vector store."),
):

    config = Config(rebuild_vector_store=rebuild)
    vector_store = build_or_load_vectorstore(config)
    chain = build_rag_chain(vector_store, config)
    print(f"Question: {question}\n")

    answer = chain.invoke(question)
    print(f"Answer: {answer}\n\n")


@app.command(help="Benchmark the RAG pipeline on the provided questions.")
def benchmark(
    llm: LLMModel = Option(
        LLMModel.GEMMA2_2B, help="Which LLM model to use for answer generation."
    ),
    embedding: EmbeddingModel = Option(
        EmbeddingModel.BGE_SMALL, help="Which embedding model to use for vector store."
    ),
    rebuild: bool = Option(True, help="Whether to rebuild the vector store."),
):
    config = Config(
        llm_model=llm,
        embedding_model=embedding,
        rebuild_vector_store=rebuild,
    )
    questions = json.load(open(config.questions_file))["questions"]

    vector_store = build_or_load_vectorstore(config)
    chain = build_rag_chain(vector_store, config)

    n_expected_sources = 0
    n_retrieved_sources = 0
    for q in questions:
        question = q["question"]
        expected_sources = q["expected_context"]

        print(f"\n\nQuestion: {question}\n")
        print(f"Expected context sources: {expected_sources}\n")
        pairs = vector_store.similarity_search_with_relevance_scores(
            question, k=config.top_k
        )
        docs = [doc for doc, _ in pairs]

        for i, (d, s) in enumerate(pairs, 1):
            print(f"[{i}] score={s:.3f} source={d.metadata.get('source')}")

        n_expected_sources += len(expected_sources)
        for doc in docs:
            if doc.metadata["source"].split("\\")[-1] in expected_sources:
                n_retrieved_sources += 1

        answer = chain.invoke(question)
        print(f"\nAnswer: {answer}\n")

    print(f"Matched {n_retrieved_sources}/{n_expected_sources} expected sources.")


if __name__ == "__main__":
    app()
