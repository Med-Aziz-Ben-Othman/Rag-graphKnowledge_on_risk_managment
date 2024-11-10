import time
from pinecone import Pinecone
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
import os
from Rag_System.data_prep.chuncking_data import refine_query_for_rag_system

def connect_to_existing_pinecone_index(index_name="ragsystemfinal", namespace="ragsystemfinal"):
    # Connect to the Pinecone index using the provided API key and index name
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    if index_name not in pc.list_indexes().names():
        raise ValueError(f"The index {index_name} does not exist in Pinecone.")
    
    # Return the Pinecone index for further use
    print(f"Connected to the existing Pinecone index: {index_name}")
    return pc.Index(index_name)

def create_retrieval_chain_from_existing_index(index, model_name="multilingual-e5-large", namespace="ragsystemfinal"):
    # Initialize the embeddings
    embeddings = PineconeEmbeddings(model=model_name, pinecone_api_key=os.getenv("PINECONE_API_KEY"))
    
    # Create the Pinecone vector store using the existing index and namespace
    docsearch = PineconeVectorStore(index_name=index, embedding=embeddings, namespace=namespace)
    
    # Set up the Langchain retrieval and LLM model
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    retriever = docsearch.as_retriever()
    
    # Initialize the OpenAI LLM
    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model_name='gpt-4o-mini',
        temperature=0.4
    )
    
    # Create the document combination chain
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    
    # Create the final retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    return retrieval_chain

def run_query_on_existing_index(query, retrieval_chain):
    # Refine the query for the RAG system
    refined_query = refine_query_for_rag_system(query)  # Ensure you have this function implemented
    print(f"Refined query: {refined_query}")
    
    # Get the answer from the retrieval chain
    answer_with_knowledge = retrieval_chain.invoke({"input": refined_query})
    
    # Output the answer and context
    print("Answer with knowledge:\n", answer_with_knowledge['answer'])
    print("\nContext used:\n", answer_with_knowledge['context'])
    return answer_with_knowledge

