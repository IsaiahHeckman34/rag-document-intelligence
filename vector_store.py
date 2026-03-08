"""
Simple numpy-based vector store with sentence-transformers embeddings.
Replaces ChromaDB for Python 3.14 compatibility.
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


class VectorStore:
    def __init__(self, store_dir: str | Path):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._embeddings_path = self.store_dir / "embeddings.npz"
        self._metadata_path = self.store_dir / "metadata.json"
        self._load()

    def _load(self):
        if self._embeddings_path.exists() and self._metadata_path.exists():
            data = np.load(self._embeddings_path)
            self.embeddings = data["embeddings"]
            with open(self._metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.ids = meta["ids"]
            self.documents = meta["documents"]
            self.metadatas = meta["metadatas"]
        else:
            self.embeddings = np.empty((0, 0), dtype=np.float32)
            self.ids = []
            self.documents = []
            self.metadatas = []

    def _save(self):
        np.savez_compressed(self._embeddings_path, embeddings=self.embeddings)
        with open(self._metadata_path, "w", encoding="utf-8") as f:
            json.dump({
                "ids": self.ids,
                "documents": self.documents,
                "metadatas": self.metadatas,
            }, f)

    def clear(self):
        self.embeddings = np.empty((0, 0), dtype=np.float32)
        self.ids = []
        self.documents = []
        self.metadatas = []
        for p in (self._embeddings_path, self._metadata_path):
            p.unlink(missing_ok=True)

    def add(self, ids: list[str], documents: list[str], metadatas: list[dict]):
        model = _get_model()
        new_embeddings = model.encode(documents, show_progress_bar=False)
        new_embeddings = np.array(new_embeddings, dtype=np.float32)

        if self.embeddings.size == 0:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        self.ids.extend(ids)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self._save()

    def query(self, query_text: str, n_results: int = 5) -> list[dict]:
        if len(self.ids) == 0:
            return []

        model = _get_model()
        query_emb = model.encode([query_text], show_progress_bar=False)
        query_emb = np.array(query_emb, dtype=np.float32)[0]

        # Cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_emb)
        norms = np.maximum(norms, 1e-10)
        similarities = self.embeddings @ query_emb / norms
        # Convert to distance (lower = more similar, matching ChromaDB convention)
        distances = 1.0 - similarities

        n = min(n_results, len(self.ids))
        top_indices = np.argsort(distances)[:n]

        results = []
        for idx in top_indices:
            results.append({
                "text": self.documents[idx],
                "source": self.metadatas[idx]["source"],
                "chunk_index": self.metadatas[idx]["chunk_index"],
                "distance": float(distances[idx]),
            })
        return results

    @property
    def count(self) -> int:
        return len(self.ids)
