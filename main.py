from typing import List, Dict, Any, Optional
import ollama

from src.ingest import get_all_files, read_all_data
from src.embedder import BGEEmbedder
from src.store import QdrantVectorStore


class RetrieverService:
    """Handles query embedding retrieval and context formatting from Qdrant."""

    def __init__(self, vector_store: QdrantVectorStore, embedder: BGEEmbedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Embeds query using BGE instruction prefix and fetches top_k similar chunks.
        """
        query_embedding = self.embedder.embed_query(query)
        
        # Use vector_store search / query_points
        results = self.vector_store.client.query_points(
            collection_name=self.vector_store.collection_name,
            query=query_embedding,
            limit=top_k
        ).points

        return [
            {
                'text': r.payload['text'],
                'file_name': r.payload['file_name'],
                'page': r.payload['page'],
                'chunk_type': r.payload['chunk_type'],
                'score': r.score,
            }
            for r in results
        ]

    @staticmethod
    def format_context(chunks: List[Dict[str, Any]]) -> str:
        """Formats retrieved chunk list into a single context string."""
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

Context:
{context}

Question: {question}

Answer:"""

    def __init__(self, model_name: str = "qwen2.5:7b", temperature: float = 0.1):
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


class RAGPipeline:
    """Main orchestrator for ingestion, retrieval, and response generation."""

    def __init__(
        self,
        collection_name: str = "insurance_chunks",
        model_name: str = "BAAI/bge-small-en-v1.5",
        llm_model: str = "qwen2.5:7b"
    ):
        self.collection_name = collection_name
        self.vector_store = QdrantVectorStore(collection_name=self.collection_name)
        self.embedder = BGEEmbedder(model_name=model_name)
        
        self.retriever = RetrieverService(self.vector_store, self.embedder)
        self.llm = LLMService(model_name=llm_model)

    def run_ingestion_and_indexing(self, force_reindex: bool = False, docs_dir: str = './documents'):
        """Parses PDFs, generates BGE embeddings, and stores chunks into Qdrant."""
        existing_count = self.vector_store.get_count()
        if existing_count > 0 and not force_reindex:
            print(f"Skipping Ingestion: Collection '{self.collection_name}' contains {existing_count} points.")
            return

        print("Data missing or reindex forced. Starting PDF ingestion...")

        # 1. Gather PDF files & parse chunks
        pdf_files = get_all_files(docs_dir)
        chunks = read_all_data(pdf_files)
        total_chunk_count = len(chunks)

        if total_chunk_count == 0:
            print("No chunks extracted. Stopping execution.")
            return

        # 2. Embed Passages
        print("\n---> Generating BGE Passages Embeddings...")
        chunk_texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed_passages(chunk_texts)

        # 3. Store in Qdrant
        print("\n---> Upserting to Qdrant Vector DB...")
        self.vector_store.create_collection(vector_size=self.embedder.vector_dim, recreate=force_reindex)
        self.vector_store.upsert_chunks(chunks=chunks, embeddings=embeddings)

        # 4. Verification Check
        final_count = self.vector_store.get_count()
        print("==========================================")
        print(f"Total Chunks Processed: {total_chunk_count}")
        print(f"Qdrant DB Point Count : {final_count}")
        print("==========================================")
        assert total_chunk_count == final_count, "Point count mismatch detected!"

    def answer_question(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Runs full RAG pipeline for a user question."""
        chunks = self.retriever.retrieve_chunks(question, top_k=top_k)
        context = self.retriever.format_context(chunks)
        answer = self.llm.generate(question, context)

        return {
            'answer': answer,
            'retrieved_chunks': chunks,
        }


if __name__ == "__main__":
    # Initialize Pipeline
    pipeline = RAGPipeline(
        collection_name="insurance_chunks",
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
        result = pipeline.answer_question(q, top_k=5)
        print("\n--- Question & Answer ---")
        print(f"\nQ: {q}\nA: {result['answer']}\n")


        print("\n--- Sources Used ---")
        for c in result['retrieved_chunks']:
            print(f"[{c['file_name']} p.{c['page']} score={c['score']:.3f}]")
            print(c['text'][:400])
            print()

        # for c in result['retrieved_chunks']:
        #     print(f"• {c['file_name']} (Page {c['page']}) - Score: {c['score']:.4f}")