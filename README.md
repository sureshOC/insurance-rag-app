Markdown
# 📄 Insurance Policy RAG System

A production-ready Retrieval-Augmented Generation (RAG) pipeline for querying insurance policy documents using local LLMs and vector search. 

This system extracts text from PDF policies, embeds chunks using BAAI's BGE model, stores them in a Qdrant Vector Database, and answers domain-specific coverage questions using Ollama (`qwen2.5:7b`).

---

## 🌟 Features

- **🔒 100% Local & Private:** Runs completely on local hardware via Ollama and Qdrant—no external API keys required.
- **⚡ Asymmetric BGE Retrieval:** Uses `BAAI/bge-small-en-v1.5` with query instruction prefixes for precision retrieval.
- **📍 Precise Citation & Grounding:** Cites specific policy document names, page numbers, and vector similarity scores for every answer.
- **🛡️ Strict Guardrail Rules:** Tailored system prompts prevent hallucinations and reject out-of-scope calculation requests while preserving coverage accuracy.

---

## 🏗️ Architecture Flow

```text
[PDF Documents] ──► [PDF Ingestion] ──► [BGE Passage Embeddings] ──► [Qdrant Vector DB]
                                                                            │
[User Query]    ──► [BGE Query Embedding] ─────────────────────────────────┘
                               │
                        [Top-K Chunks]
                               │
                               ▼
                        [Formatted Context] ──► [Ollama LLM] ──► [Answer + Citations]
📁 Repository Structure
Plaintext
.
├── documents/            # Place your insurance policy PDFs here
├── src/
│   ├── __init__.py
│   ├── ingest.py        # PDF extraction & text chunking
│   ├── embedder.py      # BGE embedding model logic
│   └── store.py         # Qdrant client & vector operations
├── main.py              # Pipeline orchestrator & entry point
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
🛠️ Prerequisites
Python 3.9+ installed on your machine.

Qdrant Vector Database: Running locally (via Docker/binary) or set up via Qdrant Cloud.

Ollama: Install Ollama and pull the LLM:

Bash
ollama pull qwen2.5:7b
🚀 Quickstart Guide
1. Clone the Repository
Bash
git clone [https://github.com/your-username/insurance-policy-rag.git](https://github.com/your-username/insurance-policy-rag.git)
cd insurance-policy-rag
2. Set Up Virtual Environment
Bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
3. Install Dependencies
Bash
pip install -r requirements.txt
4. Add Documents
Place any insurance policy PDF files into the documents/ folder:

Bash
mkdir -p documents
# Copy your PDF files into ./documents/
5. Run the Pipeline
Execute the main script to ingest documents and run query tests:

Bash
python main.py
⚙️ Configuration
You can configure collection names, models, and re-indexing behavior directly inside main.py:

Python
# Initialize Pipeline
pipeline = RAGPipeline(
    collection_name="insurance_chunks",   # Qdrant collection name
    model_name="BAAI/bge-small-en-v1.5", # Embedding model
    llm_model="qwen2.5:7b"               # Ollama model
)

# Set force_reindex=True to wipe existing collection and re-embed PDFs
pipeline.run_ingestion_and_indexing(force_reindex=False, docs_dir="./documents")
📊 Example Output
Plaintext
---> Generating BGE Passages Embeddings...
---> Upserting to Qdrant Vector DB...
==========================================
Total Chunks Processed: 42
Qdrant DB Point Count : 42
==========================================

--- Question & Answer ---

Q: Does this policy cover cancer treatment?
A: Yes, cancer treatment is covered under Section 3.2 (Critical Illness Cover), subject to a 90-day waiting period from policy inception. [Source: Health_Policy_2024.pdf, Page 12]

--- Sources Used ---
[Health_Policy_2024.pdf p.12 score=0.884]
Section 3.2 - Cancer Care Benefit: The policy covers medical expenses incurred for inp

