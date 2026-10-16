def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size")

    chunks = []

    step = chunk_size - overlap

    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size]

        chunks.append(chunk)

        if start + chunk_size >= len(text):
            break

    return chunks


if __name__ == "__main__":
    from document_loader import load_document

    document = load_document("data/banking_guide.txt")

    chunks = split_text(
        document,
        chunk_size=300,
        overlap=50
    )

    print("Total chunks:", len(chunks))

    for index, chunk in enumerate(chunks):
        print(f"\n--- Chunk {index + 1} ---")
        print(chunk)