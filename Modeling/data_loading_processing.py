import json
import torch
from torch_geometric.data import Data
from utils import make_all_lowercase  # Import helper function

def load_data(file_path='Projet_AI_Cognition/output_Graph/final_graph_data.json'):
    # Load JSON data
    with open(file_path, 'r') as f:
        graph_data = json.load(f)
    
    # Convert all keys to lowercase
    graph_data = make_all_lowercase(graph_data)
    
    # Extract nodes and relationships
    nodes = graph_data['nodes']
    relationships = graph_data['relationships']
    
    # Create node index mapping
    node_to_idx = {node['name']: idx for idx, node in enumerate(nodes)}
    
    # Prepare edges and identify missing nodes
    edges = []
    missing_nodes = []
    for rel in relationships:
        if rel['source'] in node_to_idx and rel['target'] in node_to_idx:
            edges.append((node_to_idx[rel['source']], node_to_idx[rel['target']]))
        else:
            if rel['source'] not in node_to_idx:
                missing_nodes.append(rel['source'])
            if rel['target'] not in node_to_idx:
                missing_nodes.append(rel['target'])
    
    print("Missing nodes:", missing_nodes)
    
    # Convert edges to tensor
    edges_list = [[node1_index, node2_index] for node1_index, node2_index in edges]
    edge_index = torch.tensor(edges_list, dtype=torch.long).t().contiguous()
    
    # Load node features (embeddings)
    node_features = torch.tensor(graph_data['x'])  # Replace with actual embeddings array
    
    # Extract and map labels from risk criticality
    label_mapping = {"low": 0, "medium": 1, "high": 2}
    node_labels = [label_mapping.get(node.get('attributes', {}).get('risk_criticality', 'low'), 0) for node in nodes]
    node_labels = torch.tensor(node_labels)
    
    # Create PyTorch Geometric data object
    data = Data(
        x=node_features,
        edge_index=edge_index,
        y=node_labels
    )
    
    # Save data object
    torch.save(data, 'Projet_AI_Cognition/data/graph_data_object.pt')
    print("Graph data saved successfully to 'graph_data_object.pt'.")

def load_saved_data():
    # Load saved data object
    loaded_data = torch.load('Projet_AI_Cognition/data/graph_data_object.pt')
    print("Graph data loaded successfully from 'graph_data_object.pt'.")
    return loaded_data
