
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, Request
from pydantic import BaseModel, Field, field_validator

from src.rag_service import RAGService

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize resources once
    app.state.rag_service = RAGService()

    print("RAG service initialized successfully.")

    yield

    print("Shutting down RAG service...")
    # Shutdown: release application reference
    del app.state.rag_service


app = FastAPI(
    title="AI Chatbot API",
    description="RAG chatbot",
    version="0.1.0",
    lifespan=lifespan
)

class ChatRequest(BaseModel):
    question: str = Field(min_length=1)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        question = value.strip()

        if not question:
            raise ValueError("Question cannot be empty")

        return question

class ChatResponse(BaseModel):
    answer: str

def get_rag_service(request: Request) -> RAGService:
    return request.app.state.rag_service


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    service: RAGService = Depends(get_rag_service)
):
    print("Question received:", request.question)
    print("Service instance:", id(service))

    answer = service.ask(request.question)

    return ChatResponse(answer=answer)