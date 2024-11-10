from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from Rag_System.embeddings.embedding import get_embeddings
from Rag_System.embeddings.pinecone_init import setup_pinecone_index

def create_langchain_retrieval_chain():
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    
    # Initialize Pinecone index and embeddings
    pinecone_index = setup_pinecone_index()
    retriever = pinecone_index.as_retriever()
    
    llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0.0)
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    return retrieval_chain
