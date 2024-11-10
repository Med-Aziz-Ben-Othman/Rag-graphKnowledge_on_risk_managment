import pandas as pd
import sys
import os

# Adjust sys.path to include the directory where 'Llms' is located
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Risk_Managment_Project', 'Llms')))
from model import LLMModel  # Now you can directly import LLMModel

def save_to_file(content, folder, filename):
    """Save content to a specified folder and filename with UTF-8 encoding."""
    os.makedirs(folder, exist_ok=True)  # Create the folder if it doesn't exist
    file_path = os.path.join(folder, filename)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)


def load_data(file_path):
    """ Load the processes and child processes from the DataFrame """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def find_relevant_relations(df, project_details):
    """Search the DataFrame for relevant relationships based on the project details."""
    if df is None:
        print("DataFrame is None, cannot search for relevant relationships.")
        return None
    
    # Clean DataFrame by removing NaN
    df_cleaned = df.dropna(subset=['Sentence'])

    # Extract keywords from project_details for better matching
    keywords = [word for word in project_details.lower().split() if len(word) > 3]

    # Search for relevant relationships using the keywords
    relevant_relations = df_cleaned[df_cleaned['Sentence'].str.contains('|'.join(keywords), case=False, na=False)]
    
    if relevant_relations.empty:
        print("No relevant relations found.")
    else:
        print(f"Found {len(relevant_relations)} relevant relations.")
    
    return relevant_relations

def generate_prompt(project_details, relevant_relations):
    """Generate a prompt by combining project details and the relevant relationships."""
    if relevant_relations is None or relevant_relations.empty:
        return "No relevant relationships found based on the project details provided."

    prompt = f"Here are the details of the new project: {project_details}\n"
    prompt += "These are the relevant risk relationships from similar past scenarios:\n"
    
    for index, row in relevant_relations.iterrows():
        prompt += f"- {row['Noun1']} {row['Is/Have']} {row['Noun2']} (from: {row['Sentence']})\n"
    
    prompt += "Based on the above, what are the recommended next steps for mitigating risks or improving project success?"
    return prompt

def run_llm(project_details):
    data_file = 'Risk_Managment_Project/data/conceptualization.csv'  # Correct the path to data
    data_file_path = os.path.join(os.path.dirname(__file__), data_file)  # Use absolute path
    df = load_data(data_file_path)

    # If the DataFrame is None, exit early
    if df is None:
        print("Failed to load the data. Exiting.")
        return

    relevant_relations = find_relevant_relations(df, project_details)

    prompt = generate_prompt(project_details, relevant_relations)
    
    # Save the generated prompt to the 'generated prompt' folder
    save_to_file(prompt, "generated_prompt", "prompt.txt")

    model = LLMModel()
    response = model.generate_response(prompt)
    
    # Save the generated response to the 'responses' folder
    save_to_file(response, "responses", "response.txt")
    
    print(f"Prompt: {prompt}\nResponse: {response}\n")

if __name__ == "__main__":
    new_project_details = '''
        This project is simply making cereal for breakfast, including sourcing the cereal and milk, a bowl, and a spoon.

        Outcome: You’re no longer hungry, and you’re ready to start the day.

        Delineate the tasks: 
        1. Go to the supermarket
        2. Buy a bowl
        3. Buy a spoon
        4. Buy a bottle of milk
        5. Buy a box of cereal
        6. Return home
        7. Place the bowl on the counter
        8. Pour the cereal into the bowl
        9. Pour the milk into the bowl
        10. Place the spoon in the bowl, and eat

        The tasks in a project often have strong dependencies. For larger projects, these can often be harder to spot, which is why you have to spend the time mapping out each part of the process and building a timeline.
        
        Identify the players: For eating breakfast, it is probably just yourself, unless you have a son or a daughter who helps out in the morning. Risks include not having one of the supplies, or realizing that the milk has expired.

        Timeline: For breakfast, the timeline is immediate.

        Review: You may decide to use less honey tomorrow or add something new like fruit to your cereal for added vitamins (and taste).
        '''
    run_llm(new_project_details)
