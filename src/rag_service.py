from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama

from src.document_loader import load_document
from src.rag_chatbot import retrieve_documents, generate_answer


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

    def ask(self, question: str) -> str:

        context = retrieve_documents(
            question,
            self.chunks,
            self.embeddings,
            self.embedding_model
        )

        answer = generate_answer(
            question,
            context,
            self.llm
        )

        return answer