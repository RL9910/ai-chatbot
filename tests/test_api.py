
from fastapi.testclient import TestClient

from src.main import app

from src.main import get_rag_service


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_unknown_endpoint():
    response = client.get("/unknown")

    assert response.status_code == 404

class FakeRAGService:
    def ask_with_sources(self, question: str) -> dict:
        return {
            "answer": "Redis is used for caching. [Chunk 4]",
            "sources": [
                {
                    "chunk_id": 4,
                    "content": "Redis is used for caching.",
                    "score": 0.82
                }
            ],
            "latency_ms": 120.5
        }


def test_chat():
    app.dependency_overrides[get_rag_service] = (
        lambda: FakeRAGService()
    )

    try:
        response = client.post(
            "/chat",
            json={"question": "Why use Redis?"}
        )

        assert response.status_code == 200

        data = response.json()

        assert data["answer"] == (
            "Redis is used for caching. [Chunk 4]"
        )

        assert len(data["sources"]) == 1
        assert data["sources"][0]["chunk_id"] == 4
        assert data["sources"][0]["score"] == 0.82
        assert data["latency_ms"] == 120.5

    finally:
        app.dependency_overrides.clear()


def test_chat_empty_question():
    app.dependency_overrides[get_rag_service] = (
        lambda: FakeRAGService()
    )

    try:
        response = client.post(
            "/chat",
            json={"question": ""}
        )

        assert response.status_code == 422

    finally:
        app.dependency_overrides.clear()

def test_chat_whitespace_question():
    app.dependency_overrides[get_rag_service] = (
        lambda: FakeRAGService()
    )

    try:
        response = client.post(
            "/chat",
            json={"question": "    "}
        )

        assert response.status_code == 422

    finally:
        app.dependency_overrides.clear()

def test_chat_trims_question():
    received = []

    class CapturingRAGService:
        def ask_with_sources(self, question: str) -> dict:
            received.append(question)

            return {
                "answer": "Test answer",
                "sources": [],
                "latency_ms": 100.0
            }

    app.dependency_overrides[get_rag_service] = (
        lambda: CapturingRAGService()
    )

    try:
        response = client.post(
            "/chat",
            json={"question": "  Why use Redis?  "}
        )

        assert response.status_code == 200
        assert received == ["Why use Redis?"]

    finally:
        app.dependency_overrides.clear()