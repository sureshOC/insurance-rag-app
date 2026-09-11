import json
import os
import pytest
import httpx

API_BASE_URL = os.getenv("RAG_API_URL", "http://localhost:8000")

@pytest.fixture(scope="session")
def api_client():
    """Provides a synchronous HTTPX client configured for the RAG API endpoint."""
    with httpx.Client(base_url=API_BASE_URL, timeout=120.0) as client:
        yield client

@pytest.fixture(scope="session")
def golden_dataset():
    """Loads the golden dataset for parametrized testing."""
    dataset_path = os.path.join(os.path.dirname(__file__), "test_cases.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)

@pytest.fixture(scope="session", autouse=True)
def check_service_health(api_client):
    """Ensures the RAG API container is reachable before running tests."""
    try:
        response = api_client.get("/health")
        assert response.status_code == 200, "API is running but healthcheck failed."
    except httpx.ConnectError:
        pytest.fail(f"Could not connect to RAG API at {API_BASE_URL}. Ensure Docker containers are up.")