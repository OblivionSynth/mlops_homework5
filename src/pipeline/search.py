import numpy as np
from src.embeddings.encoder import TextEmbedder
import os

class SearchPipeline:
    def __init__(self, embedder=None):
        self.embedder = embedder if embedder else TextEmbedder()
        self.data = None
        self.corpus_embeddings = None
        
    def initialize(self, data_path="data/test_data.csv", embeddings_path="embeddings.pkl"):
        """Initialize the search pipeline"""
        # Check if embeddings file exists
        if os.path.exists(embeddings_path):
            try:
                # Load embeddings
                self.embedder.load_embeddings(embeddings_path)
                self.data = self.embedder.data
                self.corpus_embeddings = self.embedder.embeddings
                
                # Make sure model is loaded
                self.embedder.load_model()
                
                return True
            except Exception as e:
                print(f"Error loading embeddings: {e}")
        
        # If embeddings don't exist or loading failed, create new ones
        try:
            # Load data
            self.data = self.embedder.load_data(data_path)
            
            # Make sure model is loaded
            self.embedder.load_model()
            
            # Create embeddings
            self.corpus_embeddings = self.embedder.create_embeddings()
            
            # Save embeddings
            self.embedder.save_embeddings(embeddings_path)
            
            return True
        except Exception as e:
            print(f"Error initializing search pipeline: {e}")
            return False
    
    def search(self, query, top_k=5):
        """Search for most similar documents to the query"""
        if self.corpus_embeddings is None:
            raise ValueError("Pipeline not initialized! Call initialize() first.")
        
        # Encode query
        query_embedding = self.embedder.model.encode(query)
        
        # Calculate cosine similarity
        cos_scores = self._cosine_similarity(query_embedding, self.corpus_embeddings)
        
        # Get top_k matches
        top_results = []
        top_indices = np.argsort(-cos_scores)[:top_k]
        
        for idx in top_indices:
            top_results.append({
                'index': int(idx),
                'score': float(cos_scores[idx]),
                'text': self.data.iloc[idx]['wikipedia_excerpt']
            })
        
        return top_results
    
    def _cosine_similarity(self, query_embedding, corpus_embeddings):
        """Calculate cosine similarity between query and all corpus embeddings"""
        return np.dot(corpus_embeddings, query_embedding) / (
            np.linalg.norm(corpus_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )