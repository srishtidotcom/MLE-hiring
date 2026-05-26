from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Sequence

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from core.document import DocumentChunk


class Embedder:
    """Generate and persist chunk embeddings for hybrid retrieval."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = 512

        # GPU detection
        if torch.cuda.is_available():
            try:
                self.model = self.model.to('cuda')
                print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
            except Exception as e:
                print(f"⚠️ GPU available but failed to move model: {e}. Using CPU.")
        else:
            print("⚠️ Running on CPU (GPU driver may need update)")

    def embed_chunks(self, chunks: Sequence[DocumentChunk]) -> np.ndarray:
        """Generate embeddings and return as numpy array."""
        texts = [chunk.text for chunk in chunks]
        if not texts:
            raise ValueError("Cannot embed an empty chunk list")

        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def save_artifacts(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: np.ndarray,
        index_dir: Path,
    ) -> None:
        """Save all artifacts needed by HybridRetriever."""
        index_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = [chunk.model_dump(exclude={"embedding"}) for chunk in chunks]
        with open(index_dir / "chunks_metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        # Save embeddings
        np.save(index_dir / "embeddings.npy", embeddings)

        # Save FAISS index
        self._save_faiss_index(embeddings, index_dir)

        print(f"✅ Index artifacts saved to {index_dir}")

    def _save_faiss_index(self, embeddings: np.ndarray, index_dir: Path):
        """Save FAISS flat index."""
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings.astype(np.float32))
        faiss.write_index(index, str(index_dir / "faiss.index"))
        print(f"   - faiss.index saved ({dimension} dimensions)")

    def save_index(self, chunks: List[DocumentChunk], output_dir: str = "../index"):
        """Convenience method for build scripts."""
        embeddings = self.embed_chunks(chunks)
        self.save_artifacts(chunks, embeddings, Path(output_dir))