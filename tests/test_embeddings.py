import pytest
import os
import tempfile
import pandas as pd
import numpy as np
from src.embeddings.encoder import TextEmbedder
from src.pipeline.search import SearchPipeline

@pytest.fixture
def sample_data():
    # Create a temporary CSV file with sample data
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        sample_texts = [
            "Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.",
            "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence.",
            "Computer vision is an interdisciplinary scientific field that deals with how computers can gain understanding from digital images or videos.",
            "Deep learning is part of a broader family of machine learning methods based on artificial neural networks."
        ]
        
        df = pd.DataFrame({'wikipedia_excerpt': sample_texts})
        df.to_csv(f.name, index=False)
        yield f.name
    
    # Clean up
    os.unlink(f.name)

def test_embedder_initialization():
    embedder = TextEmbedder()
    assert embedder.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert embedder.model is None
    assert embedder.data is None
    assert embedder.embeddings is None

def test_model_loading():
    embedder = TextEmbedder()
    model = embedder.load_model()
    assert model is not None
    assert embedder.model is not None

def test_data_loading(sample_data):
    embedder = TextEmbedder()
    data = embedder.load_data(sample_data)
    assert data is not None
    assert len(data) == 4
    assert 'wikipedia_excerpt' in data.columns

def test_embedding_creation(sample_data):
    embedder = TextEmbedder()
    embedder.load_data(sample_data)
    embeddings = embedder.create_embeddings()
    
    assert embeddings is not None
    assert embeddings.shape[0] == 4  # 4 sample texts
    assert embeddings.shape[1] > 0  # Embedding dimension should be positive

def test_search_pipeline(sample_data):
    pipeline = SearchPipeline()
    success = pipeline.initialize(sample_data)
    
    assert success is True
    assert pipeline.corpus_embeddings is not None
    assert pipeline.data is not None
    
    # Make sure the model is loaded
    if pipeline.embedder.model is None:
        pipeline.embedder.load_model()
    
    # Test searching
    results = pipeline.search("What is machine learning?", top_k=2)
    
    assert len(results) == 2
    assert 'index' in results[0]
    assert 'score' in results[0]
    assert 'text' in results[0]
    
    # The first result should be the most relevant
    assert results[0]['score'] >= results[1]['score']