from fastapi.testclient import TestClient
from src.main import app
import pytest

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Wikipedia Search API"}

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert "pipeline_initialized" in response.json()

def test_search():
    # This test assumes that you have already initialized the pipeline
    response = client.post(
        "/search",
        json={"query": "What is machine learning?", "top_k": 3}
    )
    
    # Check that we get a 200 response
    assert response.status_code == 200
    
    # Check the response structure
    response_data = response.json()
    assert "results" in response_data
    
    # Check that we got the right number of results
    results = response_data["results"]
    assert len(results) <= 3  # Could be less if dataset is small
    
    # Check the structure of each result
    if results:
        result = results[0]
        assert "index" in result
        assert "score" in result
        assert "text" in result