from typing import List, Dict, Any, Optional
import ollama

from src.ingest import get_all_files, read_all_data
from src.embedder import BGEEmbedder
from src.store import QdrantVectorStore
from sentence_transformers import CrossEncoder


# UPDATE IMPORTS AT THE TOP OF THE MAIN FILE:
from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector

from typing import List, Dict, Any
from qdrant_client.models import Prefetch, SparseVector, FusionQuery, Fusion

class RetrieverService:
    """Handles query embedding retrieval and context formatting from Qdrant."""

    CLAIM_SYNONYMS = {
        "rejected": ["repudiated", "repudiation", "denied", "not payable", "excluded"],
    }

    def __init__(self, vector_store, embedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def expand_query(self, query: str) -> List[str]:
        expanded = [query]
        for term, synonyms in self.CLAIM_SYNONYMS.items():
            if term in query.lower():
                expanded += [query.lower().replace(term, syn) for syn in synonyms]
        return expanded

    def _search_hybrid(self, query: str, top_k: int = 5, prefetch_limit: int = 20):
        """Single hybrid dense+sparse RRF search — no query expansion."""
        q_dense = self.embedder.embed_query(query)

        # Compute BM25 sparse vector representation
        sparse_vec = list(self.vector_store.sparse_model.embed([query]))[0]
        q_sparse = SparseVector(
            indices=sparse_vec.indices.tolist(),
            values=sparse_vec.values.tolist()
        )

        response = self.vector_store.client.query_points(
            collection_name=self.vector_store.collection_name,
            prefetch=[
                Prefetch(query=q_dense, using="dense", limit=prefetch_limit),
                Prefetch(query=q_sparse, using="sparse", limit=prefetch_limit),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
        )
        return response.points

    def retrieve_chunks(self, query: str, top_k: int = 5, use_expansion: bool = False) -> List[Dict[str, Any]]:
        """Default path (use_expansion=False).

        Set use_expansion=True only if hybrid alone proves insufficient.
        """
        if use_expansion:
            queries = self.expand_query(query)
            all_results = {}
            for q in queries:
                for r in self._search_hybrid(q, top_k):
                    if r.id not in all_results or r.score > all_results[r.id].score:
                        all_results[r.id] = r
            points = sorted(all_results.values(), key=lambda r: r.score, reverse=True)[:top_k]
        else:
            points = self._search_hybrid(query, top_k)

        return [{
            'text': r.payload['text'],
            'file_name': r.payload['file_name'],
            'page': r.payload['page'],
            'chunk_type': r.payload['chunk_type'],
            'chunk_id': r.payload.get('chunk_id'),
            'score': r.score,
        } for r in points]

    @staticmethod
    def format_context(chunks: List[Dict[str, Any]]) -> str:
        blocks = []
        for c in chunks:
            blocks.append(f"[Source: {c['file_name']}, Page {c['page']}]\n{c['text']}")
        return "\n\n---\n\n".join(blocks)


class LLMService:
    """Encapsulates Ollama model interactions and prompt formatting."""

    SYSTEM_PROMPT = """You are an insurance policy assistant. Answer ONLY using the provided context chunks below.

Rules:
1. If the context describes what IS or IS NOT covered, an exclusion, a definition, a condition, or a process — answer directly and cite source (file name, page number). This includes yes/no coverage questions like "does this cover X" — answer them from the context.
2. ONLY refuse when the question asks you to CALCULATE or ESTIMATE a specific rupee amount for an individual's personal claim (e.g. "how much can I claim", "what will I get paid") AND no explicit number for that exact scenario is stated in the context. In that case say: "I can't calculate this from the policy document — it depends on your specific claim details. Please contact [insurer] or your TPA."
3. Do not refuse coverage/exclusion/eligibility questions just because the topic sounds personal or medical. Only rule 2's calculation case triggers a refusal.
4. Never use outside knowledge — only what's in the context.
5. Each source file (file_name) is a SEPARATE, independent insurance policy from a different insurer. Never merge, blend, or present clauses from different source files as if they belong to one policy. If the retrieved context spans multiple different source files, group your answer explicitly by source file (e.g. "Under [file A]: ...", "Under [file B]: ..."), and if the question implies a single policy but the context only supports an answer spanning multiple unrelated files, say so explicitly rather than presenting a unified answer.
6. When citing pages, list only the exact page numbers present in the context (e.g. "Page 4, Page 6") — never state or imply a continuous range (e.g. never write "Pages 4-6") unless every page in that range actually appears in the context.
7. If the context contains multiple distinct relevant points (e.g. multiple exclusions, multiple conditions), include ALL of them in your answer — do not silently omit any relevant point present in the retrieved context.

Context:
{context}

Question: {question}

Answer:"""

    def __init__(self, model_name: str = "qwen2.5:7b", temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature

    def generate(self, question: str, context: str) -> str:
        """Sends formatted prompt to Ollama LLM and returns the text response."""
        prompt = self.SYSTEM_PROMPT.format(context=context, question=question)

        response = ollama.chat(
            model=self.model_name,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': self.temperature}
        )
        return response['message']['content']


class RerankerService:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    # def rerank(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    #     pairs = [(query, c['text']) for c in chunks]
    #     scores = self.model.predict(pairs)
    #     for c, s in zip(chunks, scores):
    #         c['rerank_score'] = float(s)
    #     return sorted(chunks, key=lambda c: c['rerank_score'], reverse=True)[:top_k]

    def rerank(self, query, chunks, top_k=5, blend_weight=0.7):
        pairs = [(query, c['text']) for c in chunks]
        rerank_scores = self.model.predict(pairs)
        # normalize both signals to comparable ranges before blending
        for c, rs in zip(chunks, rerank_scores):
            c['rerank_score'] = float(rs)
            c['final_score'] = blend_weight * c['rerank_score'] + (1 - blend_weight) * (c['score'] * 10)  # scale RRF up
        return sorted(chunks, key=lambda c: c['final_score'], reverse=True)[:top_k]

class RAGPipeline:
    """Main orchestrator for ingestion, retrieval, and response generation."""

    def __init__(
        self,
        collection_name: str = "insurance_chunks_hybrid",
        model_name: str = "BAAI/bge-small-en-v1.5",
        llm_model: str = "qwen2.5:7b"
    ):
        self.collection_name = collection_name
        self.vector_store = QdrantVectorStore(collection_name=self.collection_name)
        self.embedder = BGEEmbedder(model_name=model_name)

        self.retriever = RetrieverService(self.vector_store, self.embedder)
        self.llm = LLMService(model_name=llm_model)
        self.reranker = RerankerService()

    def run_ingestion_and_indexing(
        self,
        force_reindex: bool = False,
        docs_dir: str = './documents',
        source_collection: str = "insurance_chunks"
    ):
        """Checks for existing collections, migrates from standard to hybrid if needed,

        or ingests PDFs from scratch.
        """
        existing_count = self.vector_store.get_count()

        # Case 1: Target hybrid collection already populated
        if existing_count > 0 and not force_reindex:
            print(f"Skipping Ingestion: Collection '{self.collection_name}' contains {existing_count} points.")
            return

        # Case 2: Source collection exists -> Perform instant migration without re-embedding dense vectors
        if self.vector_store.collection_exists(source_collection) and not force_reindex:
            source_count = self.vector_store.get_count(source_collection)
            print(f"Found existing collection '{source_collection}' ({source_count} points). Migrating to hybrid...")
            self.vector_store.migrate_to_hybrid(
                source_collection=source_collection,
                expected_count=source_count,
                vector_size=self.embedder.vector_dim
            )
            return

        # Case 3: Fresh start -> Parse PDFs, encode dense + sparse, and store
        print("Data missing or reindex forced. Starting full PDF ingestion...")

        pdf_files = get_all_files(docs_dir)
        chunks = read_all_data(pdf_files)
        total_chunk_count = len(chunks)

        if total_chunk_count == 0:
            print("No chunks extracted. Stopping execution.")
            return

        print("\n---> Generating BGE Passages Embeddings...")
        chunk_texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed_passages(chunk_texts)

        print("\n---> Upserting to Hybrid Qdrant Vector DB...")
        self.vector_store.create_collection(vector_size=self.embedder.vector_dim, recreate=force_reindex)
        self.vector_store.upsert_chunks(chunks=chunks, embeddings=embeddings)

        final_count = self.vector_store.get_count()
        print("==========================================")
        print(f"Total Chunks Processed: {total_chunk_count}")
        print(f"Qdrant DB Point Count : {final_count}")
        print("==========================================")
        assert total_chunk_count == final_count, "Point count mismatch detected!"

    def answer_question(self, question: str, top_k: int = 5, retrieve_k: int = 15, use_expansion: bool = True) -> Dict[str, Any]:
        """Runs full RAG pipeline for a user question."""
        chunks = self.retriever.retrieve_chunks(question, top_k=retrieve_k, use_expansion=use_expansion)
        chunks = self.reranker.rerank(question, chunks, top_k=top_k)
        context = self.retriever.format_context(chunks)
        answer = self.llm.generate(question, context)

        return {
            'answer': answer,
            'retrieved_chunks': chunks,
        }


if __name__ == "__main__":
    # Initialize Pipeline
    pipeline = RAGPipeline(
        collection_name="insurance_chunks_hybrid",
        model_name="BAAI/bge-small-en-v1.5",
        llm_model="qwen2.5:7b"
    )

    # Step 1: Run Ingestion (will skip automatically if data exists)
    pipeline.run_ingestion_and_indexing(force_reindex=False)

    test_queries = [
        "What are the exclusions in this policy during claim?",
        "In which cases might this policy be rejected?",
        "Does this policy cover cancer treatment?",
    ]

    # Step 2: Query Answer
    for q in test_queries:
        result = pipeline.answer_question(q, top_k=5, retrieve_k=15)
        print("\n--- Question & Answer ---")
        print(f"\nQ: {q}\nA: {result['answer']}\n")


        print("\n--- Sources Used ---")
        for c in result['retrieved_chunks']:
            print(f"[{c['file_name']} p.{c['page']} score={c['score']:.3f}]")
            print(c['text'])
            print()

        # chunks_before = pipeline.retriever.retrieve_chunks(q, top_k=15, use_expansion=True)
        # chunks_after = pipeline.reranker.rerank(q, chunks_before, top_k=5)

        # print("Before rerank:")
        # for c in chunks_before[:5]: print(f"{c['file_name']} p.{c['page']} rrf={c['score']:.3f}")
        # print("After rerank (blended):")
        # for c in chunks_after: print(f"{c['file_name']} p.{c['page']} rerank={c['rerank_score']:.3f} final={c['final_score']:.3f}")

        # hybrid_only = pipeline.retriever.retrieve_chunks(q, top_k=5, use_expansion=False)
        # hybrid_plus_expansion = pipeline.retriever.retrieve_chunks(q, top_k=5, use_expansion=True)

        # print("--- Hybrid only ---")
        # for c in hybrid_only:
        #     print(f"{c['file_name']} p.{c['page']} {c['score']:.3f}")

        # print("\n--- Hybrid + expansion ---")
        # for c in hybrid_plus_expansion:
        #     print(f"{c['file_name']} p.{c['page']} {c['score']:.3f}")
        #     print(c.get('chunk_id'), c['file_name'], c['page'])