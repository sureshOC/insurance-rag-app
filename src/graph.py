import logging
import re
from typing import Any, Dict, List, Optional, TypedDict
from langgraph.graph import END, StateGraph

from src.rag import RAGPipeline

logger = logging.getLogger("insurance-rag-graph")

# Empirically tuned threshold based on top chunk reranker cross-encoder score
RELEVANCE_THRESHOLD = 0.15

# ---------------------------------------------------------------------------
# Module-level Initialization (Executed ONCE on import)
# ---------------------------------------------------------------------------
pipeline = RAGPipeline(collection_name="insurance_chunks_hybrid",
                       model_name="BAAI/bge-small-en-v1.5",
                       llm_model="qwen2.5:7b")


# ---------------------------------------------------------------------------
# State Schema
# ---------------------------------------------------------------------------
class RAGState(TypedDict):
    question: str
    retrieved_chunks: List[Dict[str, Any]]
    reranked_chunks: List[Dict[str, Any]]
    needs_refusal: bool
    answer: str
    citations: List[Dict[str, Any]]
    out_of_domain: bool


# ---------------------------------------------------------------------------
# Node Definitions
# ---------------------------------------------------------------------------
def retrieve_node(state: RAGState) -> RAGState:
    chunks = pipeline.retriever.retrieve_chunks(state["question"], top_k=15, use_expansion=True)
    return {**state, "retrieved_chunks": chunks}

def rerank_node(state: RAGState) -> RAGState:
    retrieved = state.get("retrieved_chunks", [])
    if not retrieved:
        return {**state, "reranked_chunks": []}

    reranked = pipeline.reranker.rerank(state["question"], retrieved, top_k=5)
    return {**state, "reranked_chunks": reranked}

def relevance_check_node(state: RAGState) -> RAGState:
    chunks = state.get('reranked_chunks', [])
    top_score = chunks[0]['final_score'] if chunks else -999
    out_of_domain = top_score < RELEVANCE_THRESHOLD  # tune this empirically, see below
    return {**state, 'out_of_domain': out_of_domain}

def out_of_domain_node(state: RAGState) -> RAGState:
    return {**state, 'answer': "I don't have information about this in the insurance policy documents I have access to.", 'citations': []}

def refusal_check_node(state: RAGState) -> RAGState:
    """Heuristic check for unanswerable/calculation questions BEFORE hitting LLM."""
    # Flexible keyword/regex signals
    calc_patterns = [
        r"calculate",
        r"payout",
        r"how much.*(claim|get|receive|paid)",
        r"what will i (get|receive)",
        r"my bill is",
        r"reimburse.*amount"
    ]
    q_lower = state["question"].lower().strip()

    # Refuse if explicitly asking for claim calculation
    needs_refusal = any(re.search(pattern, q_lower) for pattern in calc_patterns)

    return {**state,
            "needs_refusal": needs_refusal,
            "retrieved_chunks": [],  # Clear retrieved chunks on hard refusal
            "answer": "I can't calculate exact claim payouts. Please contact your insurer directly."
            }


def generate_node(state: RAGState) -> RAGState:
    chunks = state.get("reranked_chunks", [])

    # Edge Case: Handle empty retrieval / out-of-scope queries gracefully
    if not chunks:
        logger.warning(f"No relevant context found for question='{state['question']}'")
        return {
            **state,
            "answer": (
                "I could not find any relevant information in the available policy documents "
                "to answer your question. Please verify your query or consult your policy schedule."
            ),
            "needs_refusal": True,
        }

    context = pipeline.retriever.format_context(chunks)
    answer = pipeline.llm.generate(state["question"], context)
    return {**state, "answer": answer, "retrieved_chunks": chunks}


def refuse_node(state: RAGState) -> RAGState:
    return {
        **state,
        "answer": (
            "I cannot calculate exact payout or claim amounts from the general policy terms. "
            "Specific claim payments depend on your deductibles, co-pays, policy limits, and actual medical bills. "
            "Please contact your insurer or TPA with your claim details."
        ),
    }

import re

_TOKEN = re.compile(
    r"([\w\-\.]+\.pdf)|Pages?\s*(\d+(?:\s*,\s*(?:Page\s*)?\d+)*)", re.IGNORECASE
)

def extract_citation_pairs(answer_text: str):
    pairs, current_file = set(), None
    for m in _TOKEN.finditer(answer_text):
        if m.group(1):
            current_file = m.group(1).split("/")[-1]
        elif m.group(2) and current_file:
            for p in re.findall(r"\d+", m.group(2)):
                pairs.add((current_file, p))
    return pairs

def format_citations_node(state: RAGState) -> RAGState:
    if state.get("needs_refusal", False) or state.get("out_of_domain", False) or not state.get("answer"):
        return {**state, "citations": []}

    answer_text = state["answer"]
    reranked = state.get("reranked_chunks", [])

    # Extract (file, page) PAIRS from the answer text, not two independent sets.
    # Matches patterns like "documents/foo.pdf, Page 2" or "foo.pdf, Page 2"
    cited_pairs = set(re.findall(r"([\w\-\.]+\.pdf),?\s*Page\s*(\d+)", answer_text, re.IGNORECASE))
    # normalize to (filename_only, page_as_str)
    cited_pairs = extract_citation_pairs(answer_text)

    seen_citations = set()
    citations = []

    for c in reranked:
        clean_filename = c["file_name"].split("/")[-1]
        cite_key = (clean_filename, str(c["page"]))
        if cite_key in cited_pairs and cite_key not in seen_citations:
            seen_citations.add(cite_key)
            citations.append({"file_name": clean_filename, "page": c["page"]})

    return {**state, "citations": citations}


# ---------------------------------------------------------------------------
# Graph Compilation
# ---------------------------------------------------------------------------
workflow = StateGraph(RAGState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("rerank", rerank_node)
workflow.add_node("check_refusal", refusal_check_node)
workflow.add_node("check_relevance", relevance_check_node)
workflow.add_node("generate", generate_node)
workflow.add_node("refuse", refuse_node)
workflow.add_node("out_of_domain", out_of_domain_node)
workflow.add_node("format_citations", format_citations_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "rerank")
workflow.add_edge("rerank", "check_refusal")

# Tri-branch routing logic
def route_next(state: RAGState) -> str:
    if state["needs_refusal"]:
        return "refuse"
    return "check_relevance"

def route_relevance(state: RAGState) -> str:
    if state["out_of_domain"]:
        return "out_of_domain"
    return "generate"

workflow.add_conditional_edges(
    "check_refusal",
    route_next,
    {"refuse": "refuse", "check_relevance": "check_relevance"},
)

workflow.add_conditional_edges(
    "check_relevance",
    route_relevance,
    {"out_of_domain": "out_of_domain", "generate": "generate"},
)

workflow.add_edge("generate", "format_citations")
workflow.add_edge("refuse", END)
workflow.add_edge("out_of_domain", END)
workflow.add_edge("format_citations", END)

# Compiled LangGraph Application exposed at module scope
app = workflow.compile()


# ---------------------------------------------------------------------------
# Standalone Execution Verification
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n--- Test 1: Calculation Query (Should Refuse) ---")
    res1 = app.invoke(
        {"question": "How much can I claim now for my surgery?"}
    )
    print("Answer        :", res1["answer"])
    print("Needs Refusal :", res1["needs_refusal"])
    print("Citations     :", res1["citations"])

    print("\n--- Test 2: Standard Coverage Query ---")
    res2 = app.invoke({"question": "Does this policy cover cancer treatment?"})
    print("Answer        :", res2["answer"])
    print("Needs Refusal :", res2["needs_refusal"])
    print("Citations     :", res2["citations"])

    print("\n--- Test 3: Out-of-Scope Query ---")
    res3 = app.invoke({"question": "What is the capital of France?"})
    print("Answer        :", res3["answer"])
    print("Needs Refusal :", res3["needs_refusal"])
    print("Citations     :", res3["citations"])