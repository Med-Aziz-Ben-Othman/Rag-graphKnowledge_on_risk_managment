import uuid
import openai
import os 

class Document:
    def __init__(self, id, page_content, metadata=None):
        self.id = id
        self.page_content = page_content
        self.metadata = metadata or {}

def generate_chunks_with_gpt4(text, model="gpt-4o-mini", max_tokens=10000):
    """Generate semantically meaningful chunks using GPT-4 Mini by splitting the input text."""
    # Split the text into smaller chunks if it's too large
    chunks = [text[i:i+max_tokens] for i in range(0, len(text), max_tokens)]

    all_chunks = []
    for chunk in chunks:
        prompt = (
            "Split the following text into logical sections that preserve the meaning and flow. "
            "Each section should cover one main idea, similar to book sections. "
            "Remove any titles or headings that do not provide content or value. "
            "Here is the text:\n\n" + chunk
        )

        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "You are an expert in structuring and summarizing text."},
                      {"role": "user", "content": prompt}]
        )

        # Collect the chunks from the model response
        chunked_text = response.choices[0].message.content.strip()
        all_chunks.extend(chunked_text.split("\n\n"))

    return all_chunks

def save_refined_query(output_folder, refined_query):
    # Ensure the folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Count the existing files in the folder to determine the next index
    index = len(os.listdir(output_folder)) + 1
    output_file_path = os.path.join(output_folder, f"refined_query_{index}.txt")
    
    # Save the refined query to the new file
    with open(output_file_path, "w", encoding="utf-8") as file:
        file.write(refined_query)
        
    print(f"Refined query saved to {output_file_path}")

def refine_query_for_rag_system(original_query, model="gpt-4o-mini"):
    """Generate a refined query from the original user input to focus on Risk Management and Risk Identification."""
    prompt = (
        "Given the following query, generate a refined query that will help a Retrieval-Augmented Generation (RAG) system focus exclusively on the **Risk Management** and **Risk Identification** aspects of the project. "
        "The query should include keywords and topics that directly lead to risk management and risk identification information only. "
        "Do not include any other sections or topics in the query.\n\n"
        f"Original Query: {original_query}"
    )

    response = openai.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are an expert in summarizing and refining queries for risk-focused information retrieval."},
                  {"role": "user", "content": prompt}]
    )

    refined_query = response.choices[0].message.content.strip()
    
    # Save the refined query in the `refined_query` folder
    save_refined_query("refined_query", refined_query)
    
    return refined_query

def process_pdf_with_gpt4(text, book_name):
    """Process a cleaned text by generating chunks using GPT-4 and formatting them with book name in metadata."""
    chunks = generate_chunks_with_gpt4(text)
    documents = []

    chunk_count = 1  # Keep track of the chunk order for structured IDs
    for chunk in chunks:
        # Skip empty or heading-only chunks
        if not chunk.strip() or chunk.strip().isupper():
            continue

        # Create a structured ID for each chunk (from 1 to n)
        document_id = f"chunk_{chunk_count}"

        # Add book name to metadata
        metadata = {"book_name": book_name}

        document = Document(id=document_id, page_content=chunk, metadata=metadata)
        documents.append(document)

        chunk_count += 1  # Increment the chunk count for the next chunk

    return documents

