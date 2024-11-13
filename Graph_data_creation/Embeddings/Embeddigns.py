import torch
import json
import sys
import os
from tqdm import tqdm  # Import tqdm for progress bar

# Add the LlmDataExtractor class to the path (adjust this import if necessary)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Llms')))
from Embeddings_model import Embeddings

# Initialize the Embeddings instance
emb = Embeddings()

# Load the JSON data
with open('output_inference/project_description_data_extracted.json', 'r') as f:
    graph_data = json.load(f)

nodes = graph_data['nodes']
total_nodes = len(nodes)  # Total number of nodes for tracking progress
# Generate embeddings for each node based on the name and attributes
node_features = []
node_labels = []  # Store risk criticality labels
label_mapping = {"Low": 0, "Medium": 1, "High": 2}  # Convert textual labels to numeric

# Use tqdm for a progress bar
for i, node in enumerate(tqdm(nodes, desc="Generating embeddings", unit="node")):
    node_name = node['name']
    attributes = node.get('attributes', {})
    
    # Concatenate node name and attributes for embedding
    node_text = node_name + " " + ' '.join([f"{key}: {value}" for key, value in attributes.items()])
    
    # Get the embedding for the node (it will use cache if available)
    embedding = emb.get_embedding(node_text).tolist()
    
    # Add the embedding and label to lists
    node_features.append(embedding)
    
    # Get the risk criticality label (convert to numeric using label_mapping)
    risk_criticality = attributes.get('risk_criticality', 'Low')  # Default to "Low"
    node_labels.append(label_mapping[risk_criticality])

# Convert node features and labels to tensors
node_features = torch.tensor(node_features)
node_labels = torch.tensor(node_labels)

# Save features and labels for further use
graph_data['x'] = node_features.tolist()  # Convert tensor back to list for JSON compatibility
graph_data['y'] = node_labels.tolist()

# Save the updated graph data for future use
with open('output_inference/project_desc+embeddings.json', 'w') as f:
    json.dump(graph_data, f, indent=4)

print("Embeddings generated and saved successfully.")
