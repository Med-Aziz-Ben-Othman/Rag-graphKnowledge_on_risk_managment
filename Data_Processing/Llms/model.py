import openai
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Constants.config import config  # Ensure the correct import path

class LLMModel:
    def __init__(self):
        config_instance = config()  # Renaming to avoid conflict
        config_instance.setup_openai()  # Setup OpenAI API
    def generate_response(self, prompt):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",  # Choose the appropriate OpenAI model
                messages=[
                    {"role": "system", "content": '''You are a highly skilled assistant specialized in project risk management analysis. You are tasked with analyzing datasets that represent conceptual graphs, where nodes indicate specific entities (such as tasks, milestones, or risks) and relations define the connections between them (such as dependencies, influences, or causal links). Your role is to:

    Identify key project risk management processes from the dataset by analyzing the nodes and relations.
    Extract relevant relationships that may represent risks, risk mitigation strategies, or dependencies that could affect project outcomes.
    Provide insights into potential risks or challenges based on the graph's structure and offer suggestions for managing or mitigating these risks effectively.
    Focus on extracting meaningful insights that contribute to a better understanding of the project's risk landscape and do not exceed the limits of length for GPT models.'''},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,  # Control creativity
            )
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"Error: {str(e)}")
            return None
