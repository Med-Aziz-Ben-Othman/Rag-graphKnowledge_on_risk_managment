from langchain_pinecone import PineconeEmbeddings
from Constants.config import config

# Load configuration
config = config()

# Set up embeddings with the specified model
def get_embeddings(model_name="multilingual-e5-large"):
    embeddings = PineconeEmbeddings(
        model=model_name,
        pinecone_api_key=config.pinecone_api_key
    )
    return embeddings
