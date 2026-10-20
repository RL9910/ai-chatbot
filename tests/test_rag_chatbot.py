import numpy as np

from src.rag_chatbot import retrieve_documents

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.rag_chatbot import generate_answer


class FakeEmbeddingModel:
    def encode(self, text):
        return np.array([1.0, 0.0])


def test_retrieve_documents():
    chunks = [
        "Kafka handles event streaming.",
        "Redis is used for caching.",
        "PostgreSQL is a relational database."
    ]

    embeddings = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
        [0.5, 0.5]
    ])

    fake_model = FakeEmbeddingModel()

    context = retrieve_documents(
        question="Why use Redis?",
        chunks=chunks,
        embeddings=embeddings,
        embedding_model=fake_model,
        top_k=1
    )

    assert "Redis is used for caching." in context
    assert "Kafka" not in context
    assert "[Chunk 2]" in context


def test_generate_answer():
    fake_llm = FakeListChatModel(
        responses=[
            "Redis is used to cache frequently accessed information. [Chunk 1]"
        ]
    )

    context = """
    [Chunk 1]
    Redis is used to cache frequently accessed information.
    """

    answer = generate_answer(
        question="Why does the system use Redis?",
        context=context,
        llm=fake_llm
    )

    assert "Redis" in answer
    assert "[Chunk 1]" in answer