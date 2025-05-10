from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path

class TextEmbedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.data = None
        self.embeddings = None
        
    def load_model(self):
        """Load the sentence transformer model"""
        if self.model is None:
            print(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        return self.model
    
    def load_data(self, file_path="data/test_data.csv"):
        """Load data from CSV file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        self.data = pd.read_csv(file_path)
        print(f"Loaded {len(self.data)} rows from {file_path}")
        return self.data
    
    def create_embeddings(self, texts=None):
        """Generate embeddings for text data"""
        # Load model if not already loaded
        model = self.load_model()
        
        # Use provided texts or data from CSV
        if texts is None:
            if self.data is None:
                raise ValueError("No data loaded! Call load_data() first.")
            texts = self.data['wikipedia_excerpt'].tolist()
        
        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} texts...")
        self.embeddings = model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings with shape: {self.embeddings.shape}")
        
        return self.embeddings
    
    def save_embeddings(self, file_path="embeddings.pkl"):
        """Save embeddings and data to pickle file"""
        if self.embeddings is None:
            raise ValueError("No embeddings to save! Call create_embeddings() first.")
        
        data_to_save = {
            'data': self.data,
            'embeddings': self.embeddings
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(data_to_save, f)
            
        print(f"Saved embeddings to {file_path}")
        return True
    
    def load_embeddings(self, file_path="embeddings.pkl"):
        """Load embeddings and data from pickle file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            # Fix: change pickle.dump to pickle.load
            data = pickle.load(f)
            self.data = data['data']
            self.embeddings = data['embeddings']
        
        # Make sure model is loaded
        self.load_model()
            
        print(f"Loaded embeddings with shape: {self.embeddings.shape}")
        return self.embeddings