from src.rag import RAGPipeline

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