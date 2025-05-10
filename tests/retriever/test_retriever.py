from src.retriever.retriever import get_similar_responses

def test_get_similar_responses():
    assert get_similar_responses("What is the capital of France?") == ["These are test responses"]