# llm_data_extractor.py
import sys
import os
import json
import openai
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Constants.config import config

class Embeddings:
    def __init__(self):
        config_instance = config()  # Renaming to avoid conflict
        config_instance.setup_openai()  # Setup OpenAI API
        
        # Embeddings cache file
        self.cache_file = 'embeddings_cache.json'
        
        # Load existing cache if available
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                self.embeddings_cache = json.load(f)
        else:
            self.embeddings_cache = {}

    def get_embedding(self, text):
        # Check if embedding is already in cache
        if text in self.embeddings_cache:
            return torch.tensor(self.embeddings_cache[text])
        
        # If not, generate embedding using OpenAI API
        response = openai.embeddings.create(
            input=text,
            model="text-embedding-3-small"  # Use OpenAI's latest embedding model
        )
        embedding = response.data[0].embedding
        
        # Cache the result
        self.embeddings_cache[text] = embedding
        
        # Save cache to file
        with open(self.cache_file, 'w') as f:
            json.dump(self.embeddings_cache, f)
        
        return torch.tensor(embedding)
