import pandas as pd
import os
import sys
from tqdm import tqdm  # Import tqdm for progress visualization
import spacy
import json

# Load the Spacy model for later NLP use (if needed)
nlp = spacy.load('en_core_web_lg')

# Add the LlmDataExtractor class to the path (adjust this import if necessary)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Risk_Managment_Project', 'Llms')))
from Risk_Managment_Project.Llms.data_extractor_refined import LlmDataExtractor  # Import the LlmDataExtractor class
def extract_conceptual_graph_data(df):
    """
    Extract nodes, relationships, and attributes from the DataFrame for GCN model and save LLM responses.
    
    Args:
        df (pd.DataFrame): DataFrame containing sentences from the PMI PDF book.
        
    Returns:
        list: A list of dictionaries with nodes, relationships, and attributes for each sentence.
    """
    extractor = LlmDataExtractor()
    structured_data = []

    # Ensure 'respons' folder exists for storing LLM responses
    output_folder = 'respons'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop through each sentence in the DataFrame with tqdm progress bar
    for idx, sentence in tqdm(enumerate(df['sentence']), desc="Processing sentences", total=3):
        prompt = f"Analyze the following sentence and extract nodes, relationships, and attributes related to project management risks:\n\n{sentence}"
        
        # Get response from LLM
        response = extractor.generate_response(prompt)
        print(f"Response for sentence {idx}: '{sentence}'\n{response}\n")
        
        # Save the LLM response to a file in the 'respons' folder
        if response:
            response_file = os.path.join(output_folder, f'response_{idx}.txt')
            with open(response_file, 'w', encoding='utf-8') as file:
                file.write(response)
            
            try:
                # Process the response as JSON (since the LLM is expected to return structured JSON now)
                json_response = json.loads(response)
                
                # Extract nodes, relationships, and attributes from the JSON response
                nodes = json_response.get("nodes", [])
                relationships = json_response.get("relationships", [])
                
                # Append the structured data for this sentence to the list
                structured_data.append({
                    'nodes': nodes,
                    'relationships': relationships,
                })

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON for sentence {idx}: {e}")
                # Optionally, save the problematic response for debugging
                with open(f'response_{idx}_error.txt', 'w', encoding='utf-8') as error_file:
                    error_file.write(response)

    return structured_data


def main():
    # Load your DataFrame (assuming it's stored in a CSV for this example)
    df_pmi = pd.read_csv('Risk_Managment_Project/data/df_pmi_sent.csv')  # Adjust path as necessary
    docPMI = nlp(". ".join(df_pmi.sentence))  # You can still use SpaCy for further NLP processing if needed
    
    # Extract conceptual graph data
    conceptual_graph_data = extract_conceptual_graph_data(df_pmi)
    
    # Optionally, save the structured output to a JSON file
    output_file = "gpt4o_data_sents.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(conceptual_graph_data, file, indent=4)
    print(f"\nData has been written to {output_file}")


if __name__ == "__main__":
    main()
