# Eval Results — [today's date]

Run against live FastAPI endpoint (`/query`), Qdrant + hybrid retrieval + reranking + LangGraph pipeline.

| Query | Expected | Result |
|---|---|---|
| What are the exclusions in this policy during claim? | answerable | PASS |
| In which cases might a claim be rejected? | answerable | PASS |
| Does this policy cover cancer treatment? | answerable | PASS |
| How much can I claim now for my surgery? | refusal | PASS |
| What is the capital of France? | out_of_domain | PASS |
| What is the sub-limit for cataract surgery? | refusal (no figure in corpus) | PASS |

6/6 passing. See NOTES.md for known limitations and open items not covered by this eval set.
