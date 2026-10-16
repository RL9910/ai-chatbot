from pathlib import Path

def load_document(file_path: str) -> str:
    path = Path(file_path)

    content = path.read_text(encoding="utf-8")

    return content

if __name__ == "__main__":
    document = load_document("data/banking_guide.txt")

    print(document)

    print("Number of characters:", len(document))

    missing_document = load_document("data/missing.txt")

    print(missing_document)

