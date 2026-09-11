import logging
import time
from fastapi import FastAPI, HTTPException
from api.schemas import QueryRequest, QueryResponse, HealthResponse
from src.graph import app as rag_graph, pipeline  # your compiled LangGraph app + RAGPipeline instance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("insurance-rag-api")

app = FastAPI(title="Insurance RAG Q&A API", version="0.1.0")

@app.get("/health", response_model=HealthResponse)
def health():
    try:
        qdrant_ok = pipeline.vector_store.get_count() > 0
    except Exception:
        qdrant_ok = False
    return HealthResponse(status="ok" if qdrant_ok else "degraded", qdrant_connected=qdrant_ok, llm_model=pipeline.llm.model_name)

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    start = time.time()
    try:
        result = rag_graph.invoke({"question": req.question})
    except Exception as e:
        logger.error(f"Pipeline failure for question='{req.question}': {e}")
        raise HTTPException(status_code=500, detail="Internal error while processing the question")

    latency = time.time() - start
    logger.info(f"query='{req.question}' | latency={latency:.2f}s | refusal={result['needs_refusal']} | n_citations={len(result.get('citations', []))}")

    return QueryResponse(
        answer=result["answer"],
        citations=result.get("citations", []),
        needs_refusal=result["needs_refusal"],
        retrieved_chunks=result.get("retrieved_chunks", []),
        latency=latency
    )