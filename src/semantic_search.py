from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Kafka is a distributed event streaming platform.",
    "Redis is an in-memory data store.",
    "PostgreSQL is a relational database.",
    "JWT is commonly used for authentication.",
    "Docker packages applications into containers."
]

embeddings = model.encode(documents)

print("Number of documents:", len(documents))
print("Embedding shape:", embeddings.shape)
print("First document vector:", embeddings[0][:5])


# convert a question to embedding to compare with the document embedding
# rank the similarity and return the most relevant document
question = input("\nAsk a question: ")

question_vector = model.encode(question)

print("Question embedding shape:", question_vector.shape)

scores = (
    embeddings @ question_vector
    / (
        np.linalg.norm(embeddings, axis=1)
        * np.linalg.norm(question_vector)
    )
)

print("\nSimilarity scores:")
for document, score in zip(documents, scores):
    print(f"{score:.4f} - {document}")

best_index = np.argmax(scores)

print("\nMost relevant document:")
print(documents[best_index])
print(f"Similarity: {scores[best_index]:.4f}")

# top k similar document
top_k = 3
top_indices = np.argsort(scores)[::-1][:top_k]
print(f"\nTop {top_k} relevant documents:")
for index in top_indices:
    print(f"{scores[index]:.4f} - {documents[index]}")