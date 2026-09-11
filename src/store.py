import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Any

class QdrantVectorStore:
    def __init__(self, collection_name: str = "insurance_chunks", host: str = "localhost", port: int = 6333):
        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port)

    def collection_exists(self) -> bool:
        """Check if the collection already exists in Qdrant."""
        return self.client.collection_exists(self.collection_name)

    def get_count(self) -> int:
        """Return total number of points stored in the collection."""
        if not self.collection_exists():
            return 0
        return self.client.count(collection_name=self.collection_name).count

    def create_collection(self, vector_size: int, recreate: bool = False):
        if recreate and self.collection_exists():
            self.client.delete_collection(self.collection_name)

        if not self.collection_exists():
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            print(f"Collection '{self.collection_name}' created.")

    def upsert_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]], batch_size: int = 64):
        points = []
        for chunk, vector in zip(chunks, embeddings):
            payload = {
                "text": chunk["text"],
                "file_name": chunk["file_name"],
                "page": chunk["page"],
                "chunk_type": chunk["chunk_type"],
                "clause_refs": chunk["clause_refs"],
                "method": chunk["method"],
                "chunk_id": chunk["chunk_id"],
            }
            points.append(
                PointStruct(
                    id=chunk["chunk_id"],
                    vector=vector,
                    payload=payload
                )
            )

        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(collection_name=self.collection_name, points=batch)
            print(f"Upserted batch {i // batch_size + 1} ({len(batch)} points)")