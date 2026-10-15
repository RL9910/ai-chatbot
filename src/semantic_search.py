from sentence_transformers import SentenceTransformer
import numpy as np


def calculate_similarity(document_vectors, question_vector):
    scores = (
        document_vectors @ question_vector
        / (
            np.linalg.norm(document_vectors, axis=1)
            * np.linalg.norm(question_vector)
        )
    )

    return scores


def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    documents = [
        "Kafka is a distributed event streaming platform.",
        "Redis is an in-memory data store.",
        "PostgreSQL is a relational database.",
        "JWT is commonly used for authentication.",
        "Docker packages applications into containers."
    ]

    embeddings = model.encode(documents)

    question = input("Ask a question: ")
    question_vector = model.encode(question)

    scores = calculate_similarity(embeddings, question_vector)

    top_k = 3
    top_indices = np.argsort(scores)[::-1][:top_k]

    for index in top_indices:
        print(f"{scores[index]:.4f} - {documents[index]}")


if __name__ == "__main__":
    main()

# model = SentenceTransformer("all-MiniLM-L6-v2")

# documents = [
#     "Kafka is a distributed event streaming platform.",
#     "Redis is an in-memory data store.",
#     "PostgreSQL is a relational database.",
#     "JWT is commonly used for authentication.",
#     "Docker packages applications into containers."
# ]

# embeddings = model.encode(documents)

# print("Number of documents:", len(documents))
# print("Embedding shape:", embeddings.shape)
# print("First document vector:", embeddings[0][:5])

# # convert a question to embedding to compare with the document embedding
# # rank the similarity and return the most relevant document
# question = input("\nAsk a question: ")

# question_vector = model.encode(question)

# print("Question embedding shape:", question_vector.shape)



# scores = calculate_similarity(embeddings, question_vector)

# print("\nSimilarity scores:")
# for document, score in zip(documents, scores):
#     print(f"{score:.4f} - {document}")

# best_index = np.argmax(scores)

# print("\nMost relevant document:")
# print(documents[best_index])
# print(f"Similarity: {scores[best_index]:.4f}")

# # top k similar document
# top_k = 3
# top_indices = np.argsort(scores)[::-1][:top_k]
# print(f"\nTop {top_k} relevant documents:")
# for index in top_indices:
#     print(f"{scores[index]:.4f} - {documents[index]}")