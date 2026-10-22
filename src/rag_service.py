from time import perf_counter

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama

from src.document_loader import load_document
from src.rag_chatbot import (
    retrieve_results,
    format_context,
    generate_answer
)


class RAGService:
    def __init__(self):
        document = load_document("data/banking_guide.txt")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50
        )

        self.chunks = splitter.split_text(document)

        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.embeddings = self.embedding_model.encode(
            self.chunks
        )

        self.llm = ChatOllama(
            model="gpt-oss:20b",
            temperature=0
        )

    def ask_with_sources(self, question: str) -> str:

        start_time = perf_counter()

        sources = retrieve_results(
            question,
            self.chunks,
            self.embeddings,
            self.embedding_model,
            top_k=2
        )

        context = format_context(sources)

        answer = generate_answer(
            question,
            context,
            self.llm
        )

        latency_ms = (perf_counter() - start_time) * 1000

        return {
            "answer": answer,
            "sources": sources,
            "latency" : round(latency_ms, 2)
        }

    def ask(self, question: str) -> str:
        result = self.ask_with_sources(question)
        return result["answer"]