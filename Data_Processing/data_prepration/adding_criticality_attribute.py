# main.py
import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Llms')))
from criticality_attribute import LlmDataExtractor

def generate_risk_criticality(llm_extractor, node_name, current_attributes):
    # Generate the risk criticality using OpenAI model
    risk_criticality = llm_extractor.generate_response(node_name, current_attributes)
    
    # Simplify the result to "Low", "Medium", or "High" based on the LLM output
    if risk_criticality:
        risk_criticality = risk_criticality.lower()
        if "high" in risk_criticality:
            return "High"
        elif "medium" in risk_criticality:
            return "Medium"
        elif "low" in risk_criticality:
            return "Low"
    
    return "Unknown"  # In case the LLM provides something unexpected

def main():
    # Load the JSON data
    with open('Risk_Managment_Project/data/PMBOK7.json', 'r') as f:
        graph_data = json.load(f)

    nodes = graph_data['nodes']

    # Initialize the LLM Data Extractor
    llm_extractor = LlmDataExtractor()

    # Iterate through all nodes and add the 'risk_criticality' attribute
    for node in nodes:
        node_name = node['name']
        current_attributes = node.get('attributes', {})
        
        # Use the LLM to generate the 'risk_criticality' value
        risk_criticality = generate_risk_criticality(llm_extractor, node_name, current_attributes)
        
        # Add the 'risk_criticality' attribute to the node's attributes
        current_attributes['risk_criticality'] = risk_criticality
        node['attributes'] = current_attributes  # Update the node attributes
        
        print(f"Node: {node_name}, Risk Criticality: {risk_criticality}")

    # Optionally, save the updated nodes back to a new JSON file
    with open('Risk_Managment_Project/data/updated_graph_PMBOK7.json', 'w') as f:
        json.dump(graph_data, f, indent=4)

if __name__ == "__main__":
    main()
