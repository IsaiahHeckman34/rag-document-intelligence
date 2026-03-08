import sys
from pathlib import Path

from PyPDF2 import PdfReader

from vector_store import VectorStore


DOCUMENTS_DIR = Path(__file__).parent / "documents"
STORE_DIR = Path(__file__).parent / "vector_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def ensure_documents_dir():
    DOCUMENTS_DIR.mkdir(exist_ok=True)


def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


def ingest_pdfs():
    ensure_documents_dir()

    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in the 'documents' folder.")
        print(f"Place PDF files in: {DOCUMENTS_DIR.resolve()}")
        sys.exit(1)

    store = VectorStore(STORE_DIR)
    store.clear()

    total_chunks = 0

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        text = extract_text_from_pdf(pdf_path)

        if not text.strip():
            print(f"  Skipped (no extractable text): {pdf_path.name}")
            continue

        chunks = chunk_text(text)
        print(f"  Extracted {len(chunks)} chunks")

        ids = [f"{pdf_path.stem}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": pdf_path.name, "chunk_index": i} for i in range(len(chunks))]

        store.add(ids=ids, documents=chunks, metadatas=metadatas)
        total_chunks += len(chunks)

    print(f"\nDone! Ingested {total_chunks} chunks from {len(pdf_files)} PDF(s).")
    print("You can now run query.py to ask questions.")


if __name__ == "__main__":
    ingest_pdfs()
