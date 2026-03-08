import sys
from pathlib import Path

import anthropic

from vector_store import VectorStore

STORE_DIR = Path(__file__).parent / "vector_db"
N_RESULTS = 5
MODEL = "claude-sonnet-4-6"


def get_store() -> VectorStore:
    store = VectorStore(STORE_DIR)
    if store.count == 0:
        print("No ingested documents found. Run ingest.py first.")
        sys.exit(1)
    return store


def retrieve_chunks(store: VectorStore, query: str, n_results: int = N_RESULTS) -> list[dict]:
    return store.query(query_text=query, n_results=n_results)


def ask_claude(question: str, chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}, Chunk {c['chunk_index']}]\n{c['text']}"
        for c in chunks
    )

    prompt = (
        f"Answer the following question based ONLY on the provided context. "
        f"If the context doesn't contain enough information to answer, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )

    client = anthropic.Anthropic()

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    text_blocks = [block.text for block in response.content if block.type == "text"]
    return "\n".join(text_blocks) if text_blocks else "No response received."


def main():
    store = get_store()
    print("RAG System Ready! Ask questions about your documents.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        print("\nSearching for relevant chunks...")
        chunks = retrieve_chunks(store, question)

        print(f"Found {len(chunks)} relevant chunks. Asking Claude...\n")
        answer = ask_claude(question, chunks)

        print("=" * 60)
        print("ANSWER:")
        print("=" * 60)
        print(answer)
        print("\n" + "-" * 60)
        print("SOURCES USED:")
        print("-" * 60)
        for c in chunks:
            print(f"\n[{c['source']}, Chunk {c['chunk_index']}] (distance: {c['distance']:.4f})")
            preview = c["text"][:200].replace("\n", " ")
            print(f"  {preview}...")
        print()


if __name__ == "__main__":
    main()
