import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from document_loader import load_document
from semantic_search import calculate_similarity


def main():

    # 1. load the document
    document = load_document("data/banking_guide.txt")

    # 2. Split the document with the langchain splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = splitter.split_text(document)

    # 3. Load the embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 4. Generate chunk embeddings
    chunk_embeddings = model.encode(chunks)

    print("Number of chunks:", len(chunks))

    # 5. Question 
    question = input("\nAsk a question: ")

    # 6. Question to embedding
    question_embedding = model.encode(question)

    # 7. Calculate similarity
    scores = calculate_similarity(chunk_embeddings, question_embedding)

    # 8. Retrieve top 2 chunks
    top_k = 2
    top_indices = np.argsort(scores)[::-1][:top_k]

    # 9. result
    for index in top_indices:
        print(f"\n--- Chunk {index + 1} ---")
        print(f"Similarity: {scores[index]:.4f}")
        print(chunks[index])


if __name__ == "__main__":
    main()

# document = load_document("data/banking_guide.txt")

# chunks = split_text(document, chunk_size=300, overlap=50)

# model = SentenceTransformer("all-MiniLM-L6-v2")

# chunk_embeddings = model.encode(chunks)

# print("Number of chunks:", len(chunks))
# print("Embedding shape:", chunk_embeddings.shape)


# question = input("\nAsk a question: ")

# question_embedding = model.encode(question)

# scores = calculate_similarity(
#     chunk_embeddings,
#     question_embedding
# )

# top_k = 2
# top_indices = np.argsort(scores)[::-1][:top_k]

# for index in top_indices:
#     print(f"\n--- Chunk {index + 1} ---")
#     print(f"Similarity: {scores[index]:.4f}")
#     print(chunks[index])