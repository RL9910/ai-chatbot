from document_loader import load_document
from text_splitter import split_text
from langchain_text_splitters import RecursiveCharacterTextSplitter


def main():

    document = load_document("data/banking_guide.txt")

    # custom splitter
    custom_chunks = split_text(
        document,
        chunk_size=300,
        overlap=50
    )

    # LangChain splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    langchain_chunks = splitter.split_text(document)

    # Compare results
    print("CUSTOM SPLITTER")
    print("Total chunks:", len(custom_chunks))

    for i, chunk in enumerate(custom_chunks):
        print(f"\n--- Chunk {i + 1} ---")
        print(chunk)

    print("\n\nLANGCHAIN SPLITTER")
    print("Total chunks:", len(langchain_chunks))

    for i, chunk in enumerate(langchain_chunks):
        print(f"\n--- Chunk {i + 1} ---")
        print(chunk)


if __name__ == "__main__":
    main()