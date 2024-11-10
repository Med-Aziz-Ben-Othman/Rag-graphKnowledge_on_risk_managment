import openai
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Constants.config import config  # Ensure the correct import path

class LlmDataExtractor:
    def __init__(self):
        config_instance = config()  # Renaming to avoid conflict
        config_instance.setup_openai()  # Setup OpenAI API
        
    def generate_response(self, prompt):
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",  # Choose the appropriate OpenAI model
                messages=[
                    {
                        "role": "system", 
                        "content": '''
You are a highly capable assistant focused on extracting structured data from project management text to support graph-based learning models, such as Graph Convolutional Networks (GCNs).

Your task is to analyze the provided project management text semantically, identifying key concepts, entities, and relationships between them relevant to project risk management . Structure your responses in the following format exactly and do not add anything else only in the values u can add more but the structure should be exactly this one :
{
    "nodes": [
        {
            "name": "Entity 1",
            "type": "Type of Entity", 
            "attributes": {"key1": "value1", "key2": "value2"} 
        },
        {
            "name": "Entity 2",
            "type": "Type of Entity", 
            "attributes": {"key1": "value1"}
        }
    ],
    "relationships": [
        {
            "source": "Entity 1",
            "relationship": "Relation Type",
            "target": "Entity 2"
        }
    ]
}

Ensure that:
- Nodes include relevant attributes where applicable.
- Relationships clearly connect nodes, with the type of relationship specified and explained.
- All information is well-structured and relevant to project risk management.
- Ensure in the attributes to extract all the possible attributes like synonyms type and many others.
Do not assume any predefined rules for extraction; instead, rely on your understanding of the text and the semantic relationships therein.'''
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.7,  # Control creativity
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error: {str(e)}")
            return None
