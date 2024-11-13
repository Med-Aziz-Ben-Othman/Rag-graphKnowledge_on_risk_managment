import openai
import sys
import os
import json

# Add the LlmDataExtractor class to the path (adjust this import if necessary)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from Risk_Managment_Project.Llms.data_extractor_refined import LlmDataExtractor


def extract_conceptual_graph_data(project_description):
    """
    Extract nodes, relationships, and attributes from a project description for GCN model.
    
    Args:
        project_description (str): Full text description of the project.
        
    Returns:
        dict: A dictionary containing nodes and relationships.
    """
    extractor = LlmDataExtractor()
    structured_data = {}

    # Ensure 'responses' folder exists for storing LLM responses
    output_folder = 'responses_inference'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Generate the prompt based on the entire project description
    prompt = f"Analyze the following project description and extract nodes, relationships, and attributes related to project management risks:\n\n{project_description}"

    # Get response from LLM
    response = extractor.generate_response(prompt)
    print(f"Response: '{response}'\n")

    # Save the LLM response to a file in the 'responses' folder
    if response:
        response_file = os.path.join(output_folder, 'response_project_description.txt')
        with open(response_file, 'w', encoding='utf-8') as file:
            file.write(response)

        try:
            # Process the response as JSON (since the LLM is expected to return structured JSON)
            json_response = json.loads(response)

            # Extract nodes, relationships, and attributes from the JSON response
            nodes = json_response.get("nodes", [])
            relationships = json_response.get("relationships", [])

            structured_data = {
                'nodes': nodes,
                'relationships': relationships,
            }

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            with open(f'response_error.txt', 'w', encoding='utf-8') as error_file:
                error_file.write(response)

    return structured_data


def main():
    # Load your project description (for example, it could be in a text file)
    project_description_file = 'data_inference/project_description.txt'  # Adjust the file path as necessary
    
    with open(project_description_file, 'r', encoding='utf-8') as file:
        project_description = file.read()

    # Extract conceptual graph data from the project description
    conceptual_graph_data = extract_conceptual_graph_data(project_description)
    
    # Optionally, save the structured output to a JSON file
    output_file = "output_inference/project_description_data_extracted.json"
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(conceptual_graph_data, file, indent=4)
    print(f"\nData has been written to {output_file}")


if __name__ == "__main__":
    main()
