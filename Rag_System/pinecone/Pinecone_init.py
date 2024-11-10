import time
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
import os

def setup_pinecone_index(index_name="ragsystemfinal", dimension=1024, metric="cosine"):
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    spec = ServerlessSpec(cloud=os.getenv("PINECONE_CLOUD", "aws"), region=os.getenv("PINECONE_REGION", "us-east-1"))
    
    if index_name not in pc.list_indexes().names():
        pc.create_index(name=index_name, dimension=dimension, metric=metric, spec=spec)
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
    print("Pinecone index setup complete.")
    return pc.Index(index_name)

def upsert_documents_to_index(index, documents, model_name="multilingual-e5-large", namespace="ragsystemfinal"):
    embeddings = PineconeEmbeddings(model=model_name, pinecone_api_key=os.getenv("PINECONE_API_KEY"))
    docsearch = PineconeVectorStore.from_documents(documents, index_name=index.name, embedding=embeddings, namespace=namespace)
    time.sleep(2)  # Give some time for indexing
    print("Documents upserted.")
    return docsearch

def create_retrieval_chain(docsearch):
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    retriever = docsearch.as_retriever()
    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model_name='gpt-4o-mini',
        temperature=0.4
    )
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return retrieval_chain
