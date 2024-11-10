# config/config.py
import os
import openai
import dotenv


class config:
    def __init__(self):
        self.openai_api_key = os.getenv("openai_api_key")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not self.pinecone_api_key:
            raise ValueError("Pinecone API key not found in environment variables")
    def setup_openai(self):
        openai.api_key = self.openai_api_key
        return openai
