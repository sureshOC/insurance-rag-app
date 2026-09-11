import os
import sys
from typing import Any, Dict, List
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)


class QdrantVectorStore:

    def __init__(self, collection_name: str = "insurance_chunks_hybrid", host: str = "localhost",port: int = 6333):
        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port)
        QDRANT_MODEL = "Qdrant/bm25"
        self.sparse_model = SparseTextEmbedding(model_name=QDRANT_MODEL)

    def collection_exists(self, target_collection: str | None = None) -> bool:
        """Check if a collection exists in Qdrant."""
        name = target_collection or self.collection_name
        return self.client.collection_exists(name)

    def get_count(self, target_collection: str | None = None) -> int:
        """Return total number of points stored in a collection."""
        name = target_collection or self.collection_name
        if not self.collection_exists(name):
            return 0
        return self.client.count(collection_name=name).count

    def create_collection(self, vector_size: int = 384, recreate: bool = False):
        """Creates a hybrid collection supporting both named 'dense' and 'sparse' vectors."""
        if recreate and self.collection_exists():
            self.client.delete_collection(self.collection_name)

        if not self.collection_exists():
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={"dense": VectorParams(size=vector_size, distance=Distance.COSINE)},
                sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))},
            )
            print(f"Hybrid collection '{self.collection_name}' created successfully.")

    def upsert_chunks(self, chunks: List[Dict[str, Any]],embeddings: List[List[float]],batch_size: int = 64,):
        """Upserts new chunks by computing sparse vectors on the fly alongside dense embeddings."""
        texts = [chunk["text"] for chunk in chunks]
        sparse_embeddings = list(self.sparse_model.embed(texts))

        points = []
        for chunk, dense_vector, sparse_vec in zip(chunks, embeddings, sparse_embeddings):
            payload = {
                "text": chunk["text"],
                "file_name": chunk["file_name"],
                "page": chunk["page"],
                "chunk_type": chunk["chunk_type"],
                "clause_refs": chunk["clause_refs"],
                "method": chunk["method"],
                "chunk_id": chunk["chunk_id"],
            }

            qdrant_sparse = SparseVector(indices=sparse_vec.indices.tolist(),
                                         values=sparse_vec.values.tolist())
            points.append(
                PointStruct(
                    id=chunk["chunk_id"],
                    vector={
                        "dense": dense_vector,
                        "sparse": qdrant_sparse,
                    },
                    payload=payload,
                )
            )

        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=self.collection_name, points=batch)
            print(f"Upserted batch {i // batch_size + 1} ({len(batch)} points)")

    def migrate_to_hybrid(self, source_collection: str = "insurance_chunks", expected_count: int = 822, vector_size: int = 384, batch_size: int = 100):
        """Scrolls existing points from a source collection, computes sparse vectors,

        and populates the new hybrid collection without re-parsing or re-embedding dense vectors.
        """
        print(f"Starting migration from '{source_collection}'...")

        # Ensure target hybrid collection exists
        self.create_collection(vector_size=vector_size, recreate=True)

        offset = None
        migrated_count = 0

        while True:
            records, next_offset = self.client.scroll(
                collection_name=source_collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )

            if not records:
                break

            # Batch encode sparse embeddings for retrieved text payloads
            texts = [r.payload.get("text", "") for r in records]
            sparse_embeddings = list(self.sparse_model.embed(texts))

            hybrid_points = []
            for record, sparse_vec in zip(records, sparse_embeddings):
                # Extract dense vector safely whether nested or raw list
                dense_vec = record.vector
                if isinstance(dense_vec, dict) and "dense" in dense_vec:
                    dense_vec = dense_vec["dense"]

                qdrant_sparse = SparseVector(
                    indices=sparse_vec.indices.tolist(),
                    values=sparse_vec.values.tolist(),
                )

                hybrid_points.append(
                    PointStruct(
                        id=record.id,
                        vector={
                            "dense": dense_vec,
                            "sparse": qdrant_sparse,
                        },
                        payload=record.payload,
                    )
                )

            self.client.upsert(collection_name=self.collection_name, points=hybrid_points)

            migrated_count += len(records)
            print(f"Migrated batch: {migrated_count}/{expected_count} points processed.")

            offset = next_offset
            if offset is None:
                break

        # Verification step
        final_count = self.get_count()
        print("\n" + "=" * 40)
        print(f"Source Count: {self.get_count(source_collection)}")
        print(f"Target Hybrid Count: {final_count}")
        print("=" * 40)

        if final_count == expected_count:
            print(f"SUCCESS: Successfully migrated all {expected_count} points to '{self.collection_name}'.")
        else:
            print(f"ERROR: Count mismatch! Expected {expected_count}, got {final_count}.")
            sys.exit(1)