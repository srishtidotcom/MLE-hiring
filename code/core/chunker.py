from __future__ import annotations

import hashlib
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.document import Document, DocumentChunk


class DocumentChunker:
    """Split documents into overlapping chunks for retrieval."""

    def __init__(self, chunk_size: int = 650, chunk_overlap: int = 100) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def chunk_documents(self, documents: List[Document]) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []

        for document in documents:
            texts = self.splitter.split_text(document.content)
            total_chunks = len(texts)

            for chunk_index, text in enumerate(texts):
                chunk_id = self._stable_chunk_id(document.id, chunk_index)
                metadata = dict(document.metadata)
                metadata.update(
                    {
                        "chunk_index": chunk_index,
                        "chunk_count": total_chunks,
                        "chunk_size": self.chunk_size,
                    }
                )

                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=document.id,
                        filepath=document.filepath,
                        company=document.company,
                        text=text,
                        metadata=metadata,
                    )
                )

        print(f"Created {len(chunks)} chunks from {len(documents)} documents "
              f"(size={self.chunk_size}, overlap={self.chunk_overlap})")
        return chunks

    @staticmethod
    def _stable_chunk_id(document_id: str, chunk_index: int) -> str:
        source = f"{document_id}:{chunk_index}".encode("utf-8")
        return hashlib.md5(source).hexdigest()[:16]