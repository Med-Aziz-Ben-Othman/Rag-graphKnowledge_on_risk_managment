import pandas as pd
import os
import sys
import json
from tqdm import tqdm  # Import tqdm for progress visualization
import spacy

# Load the Spacy model for later NLP use (if needed)
nlp = spacy.load('en_core_web_lg')

# Add the LlmDataExtractor class to the path (adjust this import if necessary)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Risk_Managment_Project', 'Llms')))
from Risk_Managment_Project.Llms.data_extractor_refined import LlmDataExtractor  # Import the LlmDataExtractor class

def extract_conceptual_graph_data(text):
    """
    Extract nodes, relationships, and attributes from the full text for GCN model and save LLM responses.
    
    Args:
        text (str): The entire text from the PMI PDF book as one document.
        
    Returns:
        dict: A dictionary with nodes, relationships, and attributes extracted from the entire text.
    """
    extractor = LlmDataExtractor()
    structured_data = {}

    # Ensure 'respons_all_text' folder exists for storing LLM responses
    output_folder = 'respons_all_text'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Send the whole text as a single prompt to the LLM
    prompt = f"Analyze the following text and extract nodes, relationships, and attributes related to project management risks:\n\n{text}"
    
    # Get response from LLM
    response = extractor.generate_response(prompt)
    print(f"Response for the entire text:\n{response}\n")

    # Save the LLM response to a file in the 'respons_all_text' folder
    if response:
        response_file = os.path.join(output_folder, 'response_full_text.txt')
        with open(response_file, 'w', encoding='utf-8') as file:
            file.write(response)
        
        try:
            # Process the response as JSON (since the LLM is expected to return structured JSON now)
            json_response = json.loads(response)
            
            # Extract nodes, relationships, and attributes from the JSON response
            nodes = json_response.get("nodes", [])
            relationships = json_response.get("relationships", [])
            attributes = json_response.get("attributes", {})
            
            # Store structured data
            structured_data = {
                'nodes': nodes,
                'relationships': relationships,
                'attributes': attributes
            }

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for the full text: {e}")
            # Optionally, save the problematic response for debugging
            with open(os.path.join(output_folder, 'response_full_text_error.txt'), 'w', encoding='utf-8') as error_file:
                error_file.write(response)

    return structured_data

def main():
    # Load the sentences DataFrame
    df_pmi = pd.read_csv('Risk_Managment_Project/data/df_pmi_sent.csv')  # Adjust path as necessary
    
    # Concatenate all sentences into one large text
    full_text = ". ".join(df_pmi['sentence'].tolist())
    docPMI = nlp(full_text)  # Process the full text with SpaCy if needed
    
    # Extract conceptual graph data from the entire document
    conceptual_graph_data = extract_conceptual_graph_data(docPMI.text)
    
    if conceptual_graph_data:
        # Save the structured output to a JSON file
        output_file = "gpt_4o_full_text.json"
        with open(output_file, "w", encoding="utf-8") as file:
            json.dump(conceptual_graph_data, file, indent=4)
        print(f"\nData has been written to {output_file}")
    else:
        print("No data was extracted from the text.")

if __name__ == "__main__":
    main()
