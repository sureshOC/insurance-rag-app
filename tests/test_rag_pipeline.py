import pytest
import httpx
import json

# Max acceptable end-to-end response time (seconds)
LATENCY_SLA_SECONDS = 30.0

@pytest.mark.integration
class TestRAGPipeline:

    def test_health_check(self, api_client: httpx.Client):
        """Validates that the service health endpoint returns 200 OK."""
        response = api_client.get("/health")
        assert response.status_code == 200
        assert response.json().get("status") == "ok"

    @pytest.mark.parametrize("test_case", [
        tc for tc in json.load(open("tests/test_cases.json")) if tc["category"] == "domain_knowledge"
    ], ids=lambda tc: tc["id"])
    def test_domain_queries_and_citations(self, api_client: httpx.Client, test_case: dict):
        """Validates domain queries return correct keywords, citations, and non-empty chunks."""
        payload = {"question": test_case["question"]}
        response = api_client.post("/query", json=payload)
        
        assert response.status_code == 200, f"Query failed with body: {response.text}"
        data = response.json()
        
        # 1. Assert response components
        assert "answer" in data
        assert len(data["retrieved_chunks"]) > 0, "No chunks retrieved from Qdrant"
        assert data["needs_refusal"] is False, "Domain query was wrongly flagged for refusal"
        
        # 2. Check key domain facts present in the answer
        answer_text = data["answer"].lower()
        for kw in test_case["expected_keywords"]:
            assert kw.lower() in answer_text, f"Expected keyword '{kw}' missing from answer"

        # 3. Citation validation
        if test_case["expected_filename"]:
            has_valid_citation = any(
                test_case["expected_filename"] in chunk.get("file_name", "")
                for chunk in data["retrieved_chunks"]
            )
            assert has_valid_citation, f"Expected file '{test_case['expected_filename']}' not in retrieved chunks"

    @pytest.mark.guardrails
    @pytest.mark.parametrize("test_case", [
        tc for tc in json.load(open("tests/test_cases.json")) if tc["category"] in ("guardrail_refusal", "out_of_domain")
    ], ids=lambda tc: tc["id"])
    def test_guardrails_and_refusals(self, api_client: httpx.Client, test_case: dict):
        """Validates that calculation requests and out-of-domain queries trigger refusal/relevance filters."""
        payload = {"question": test_case["question"]}
        response = api_client.post("/query", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # Assert refusal flag is toggled or context is safely restricted
        assert data["needs_refusal"] is True or len(data.get("retrieved_chunks", [])) == 0, \
            f"Query '{test_case['question']}' should have triggered refusal guardrails."

    @pytest.mark.latency
    def test_latency_sla(self, api_client: httpx.Client):
        """SLA test: verifies latency remains below the threshold."""
        payload = {"question": "What are the exclusions in this policy?"}
        response = api_client.post("/query", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        
        # Latency returned in payload (in seconds or ms depending on api implementation)
        latency = data.get("latency", 0)
        assert latency < LATENCY_SLA_SECONDS, f"Latency of {latency:.2f}s exceeded SLA threshold of {LATENCY_SLA_SECONDS}s"