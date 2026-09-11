## Day 1 — Hybrid Search (dense + sparse RRF)

### Built
- Migrated `insurance_chunks` (822 pts) → `insurance_chunks_hybrid` via scroll+re-upsert,
  no re-parsing/re-embedding of dense vectors needed. Verified 822/822.
- Added BM25 sparse vectors (fastembed `Qdrant/bm25`) alongside existing BGE dense vectors,
  fused server-side via Qdrant `FusionQuery(fusion=Fusion.RRF)` with `Prefetch`.
- Split retrieval into `_search_hybrid()` (hybrid alone) and `retrieve_chunks(use_expansion=)`
  (hybrid + synonym expansion), so the two strategies can be compared in isolation.

### Key finding — polysemy, not just lexical mismatch
Query "in which cases might a claim be rejected" needs LIC's "repudiate" clause (p.20/21).
- Hybrid-only did NOT surface it in top-5, even after widening prefetch 5→20.
- Hybrid+expansion DID surface it reliably (top scores 0.83/0.75).
- Root cause: "reject" is polysemous within the corpus itself — SBI/ICICI use it for
  claim-settlement SLA ("shall settle or reject a Claim within 15 days"), a different
  sense than "policy/claim denial." BM25 can't disambiguate word sense, so hybrid search
  actually pulled in the SLA-sense chunks over the semantically-correct repudiation clause.
  This is a real, evidenced limitation of lexical matching, not a config mistake.

### Decision
- KEEP `expand_query` as a compensating layer for known-polysemous terms in this corpus.
  Do not delete it — evidence shows hybrid-only underperforms plain dense search on this
  query, and expansion recovers it.
- RRF scores (0.833/0.500/0.250 tiers) are rank-based, not cosine similarity — not
  comparable to old dense-only scores. Note this if asked to explain a "score drop."

### Open items (not fixed tonight, carried to later)
- Prefetch-limit widening (5→20) was tested but did NOT change hybrid-only's top-5 for
  the rejected/repudiated query — expansion is still required for that case. Re-verify
  with an even wider prefetch (e.g. 50) if time allows later; not blocking.
- Eval script's `must_contain_keywords` is too rigid — a correct, well-grounded answer from
  a different insurer's policy can fail just because it doesn't match hardcoded keywords tied
  to one specific insurer's phrasing (seen tonight on the "rejected" query). Needs a looser
  or LLM-judged groundedness check eventually, not literal keyword matching.
- Q1 eval FAIL not yet root-caused — print statement truncates answers to 200 chars,
  couldn't confirm if "exclusion" keyword was actually missing or just hidden by truncation.

### Verified working
- Migration integrity: 822/822 points, dense+sparse vectors both present.
- Source-conflation, exact-page-citation, and dropped-content prompt rules (5, 6, 7) — 
  all confirmed via manual re-test on 3 core queries earlier this session.


## Day 4 — FastAPI service layer

### Built
- POST /query, GET /health, pydantic request/response schemas, structured logging (query/latency/refusal/citation count per request)
- Added a THIRD conditional branch to the LangGraph (relevance/out-of-domain check) beyond refusal — Qdrant always returns top-k regardless of true relevance, so an unrelated question (tested: "capital of France") needed its own threshold-based branch, not just the calculation-refusal one from Day 3.
- Fixed citation-matching bug: was checking file-match OR page-match as independent sets, letting an uncited file slip in if it shared a page number with a genuinely cited file. Fixed to match (file, page) as a coupled pair.

### Verified
- Empty/short input → clean 422 via pydantic validation
- Qdrant down → clean 500, logged server-side, no leaked stack trace to client
- Citations now exactly match what's textually cited in the answer, no over-inclusion

### Open items carried forward (unchanged from Day 1-3)
- blend_weight tuning (reranker vs retrieval score) still favors reranker too heavily on the "rejected/repudiated" polysemy case
- Reranker can drag in page-adjacent-but-irrelevant clauses
- Two-level table headers — documented limitation, not fixed by design
- Regex citation-matching depends on the model consistently following the exact "[Source: file, Page N]" format — a structured JSON output from the LLM would be more robust; noted as a future upgrade, not done