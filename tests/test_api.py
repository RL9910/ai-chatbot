
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
    def ask(self, question: str) -> str:
        return "Redis is used for caching. [Chunk 4]"


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
        assert response.json() == {
            "answer": "Redis is used for caching. [Chunk 4]"
        }
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
        def ask(self, question: str) -> str:
            received.append(question)
            return "Test answer"

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