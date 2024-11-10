import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from Rag_System.pinecone.connect_existing_index import connect_to_existing_pinecone_index, create_retrieval_chain_from_existing_index, run_query_on_existing_index



def save_output_to_file(output_folder, answer, context):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    output_file_path = os.path.join(output_folder, "answer_with_context_existing_index.txt")
    
    # Convert context to a string, extracting text from Document objects if needed
    context_str = "\n".join(doc.page_content for doc in context) if isinstance(context, list) else str(context)
    
    with open(output_file_path, "w", encoding="utf-8") as file:
        file.write("Answer with knowledge:\n")
        file.write(answer + "\n\n")
        file.write("Context used:\n")
        file.write(context_str)
        
    print(f"Output saved to {output_file_path}")

def main():
    """ Main function to integrate everything and handle the query execution """
    index_name = "ragsystemfinal"

    query = "What is the process for managing risks in PMI methodology?"
    
    # Connect to the existing Pinecone index
    index = connect_to_existing_pinecone_index(index_name)
    
    # Create the retrieval chain from the existing index
    retrieval_chain = create_retrieval_chain_from_existing_index(index_name)
    
    # Run a query on the existing index and get the answer
    answer_with_knowledge = run_query_on_existing_index(query, retrieval_chain)
    
    # Optionally save the output to a file
    output_folder = "output_Rag_System"
    save_output_to_file(output_folder, answer_with_knowledge['answer'], answer_with_knowledge['context'])


if __name__ == "__main__":
    main()
