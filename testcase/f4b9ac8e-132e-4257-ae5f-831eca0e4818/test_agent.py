
import pytest
from fastapi.testclient import TestClient
from agent import app

@pytest.fixture(scope="module")
def client():
    """Fixture for FastAPI test client."""
    with TestClient(app) as c:
        yield c

def test_functional_health_check_endpoint(client):
    """
    Functional test: Verifies that the /health endpoint is available and returns the expected status.
    """
    response = client.get("/health")
    assert response.status_code == 200, "Expected HTTP 200 from /health endpoint"
    data = response.json()
    assert isinstance(data, dict), "Response should be a JSON object"
    assert data.get("status") == "ok", "Expected status: ok in response"