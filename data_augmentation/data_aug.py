import json
import torch
from torch_geometric.utils import negative_sampling
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load your pre-trained model
def load_model(model_path):
    model = torch.load(model_path)
    return model

import json
import torch
from torch_geometric.data import Data

def perform_classification_and_update(inference_data, model):
    """
    Perform classification on the inference data to predict risk criticality for nodes
    and update the inference data with these predicted values.
    
    Args:
    - json_file_path (str): Path to the JSON file containing the inference data.
    - model (torch.nn.Module): The model to use for classification.
    
    Returns:
    - inference_data (dict): The updated inference data with risk criticality.
    """
    # Load the JSON data
    with open(inference_data, 'r') as f:
        inference_data = json.load(f)

    # Extract nodes, relationships, and embeddings from the loaded data
    nodes = inference_data['nodes']
    relationships = inference_data['relationships']
    embeddings = inference_data['x']

    # Convert embeddings to a PyTorch tensor
    node_features = torch.tensor(embeddings, dtype=torch.float)

    # Map node names to indices
    node_to_idx = {node['name']: idx for idx, node in enumerate(nodes)}

    # Prepare edge index based on relationships
    edges = [(node_to_idx[rel['source']], node_to_idx[rel['target']]) for rel in relationships]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Create a Data object for inference
    data = Data(x=node_features, edge_index=edge_index)

    # Perform inference
    with torch.no_grad():
        # Perform a forward pass to get node logits from the model
        node_logits = model(data.x, data.edge_index)
        
        # Get the predicted class for each node
        node_predictions = node_logits.argmax(dim=1)
        
        # Define class labels for risk criticality
        class_labels = ['Low', 'Medium', 'High']
        
        # Map predicted class to the node names
        predicted_risk_levels = {
            node['name'].lower(): class_labels[pred.item()]
            for node, pred in zip(nodes, node_predictions)
        }

    # Add predicted risk levels to the nodes in the inference data
    for node in nodes:
        node_name = node['name'].lower()
        if node_name in predicted_risk_levels:
            # Ensure 'attributes' key exists and add 'risk_criticality'
            node.setdefault('attributes', {})['risk_criticality'] = predicted_risk_levels[node_name]

    # Return the updated inference data with the new risk levels
    return inference_data



def perform_link_prediction(inference_data, source_data, model):
    """
    Perform link prediction on both source and inference data and return the suggested new links.
    """
    # Concatenate both source and inference data for link prediction
    combined_edge_index = torch.cat([source_data['edge_index'], inference_data['edge_index']], dim=1)
    
    # Generate negative samples for link prediction
    num_neg_samples = combined_edge_index.size(1)
    negative_edge_index = negative_sampling(
        edge_index=combined_edge_index,
        num_nodes=inference_data.x.size(0),
        num_neg_samples=num_neg_samples
    )

    # Prepare positive and negative pairs
    positive_pairs = combined_edge_index
    negative_pairs = negative_edge_index
    all_pairs = torch.cat([positive_pairs, negative_pairs], dim=1)
    labels = torch.cat([torch.ones(positive_pairs.size(1)), torch.zeros(negative_pairs.size(1))])

    # Forward pass to get node embeddings
    with torch.no_grad():
        model(inference_data.x, combined_edge_index)

    # Predict links for all pairs
    link_predictions = model.predict_links(all_pairs)

    # Convert predictions to binary labels using a threshold
    predicted_labels = (link_predictions > 0.4).float().squeeze()

    # Evaluate the prediction performance
    accuracy = accuracy_score(labels.cpu(), predicted_labels.cpu())
    precision = precision_score(labels.cpu(), predicted_labels.cpu())
    recall = recall_score(labels.cpu(), predicted_labels.cpu())
    f1 = f1_score(labels.cpu(), predicted_labels.cpu())

    print(f"Link Prediction - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    # Get scores for negative pairs and suggest new links
    link_scores = link_predictions.squeeze()
    negative_scores = link_scores[positive_pairs.size(1):]
    negative_pairs = all_pairs[:, positive_pairs.size(1):]

    _, top_indices = torch.topk(negative_scores, k=5)
    suggested_new_links = negative_pairs[:, top_indices]

    return suggested_new_links

def merge_data(source_data, inference_data, suggested_new_links):
    """
    Merge the source data with the inference data and add the suggested new links.
    """
    # Merge nodes from both source and inference data
    node_index_to_name = {idx: node['name'] for idx, node in enumerate(source_data['nodes'])}
    
    # Add the nodes from the inference data to the source data
    existing_node_names = {node['name'].lower() for node in source_data['nodes']}
    for node in inference_data['nodes']:
        if node['name'].lower() not in existing_node_names:
            source_data['nodes'].append(node)

    # Prepare new relationships from the suggested links
    new_links = []
    existing_relationships = {(rel['source'], rel['target']) for rel in source_data.get('relationships', [])}

    print("Top suggested new links with node names:")
    for i, (node1_idx, node2_idx) in enumerate(suggested_new_links.t()):
        node1_name = node_index_to_name.get(node1_idx.item(), "Unknown")
        node2_name = node_index_to_name.get(node2_idx.item(), "Unknown")
        print(f"Suggested link {i+1}: {node1_name} - {node2_name}")

        # Add only unique new links
        if (node1_name, node2_name) not in existing_relationships:
            new_links.append({
                "source": node1_name,
                "relationship": "suggested_link",
                "target": node2_name
            })

    # Append new links to source data
    source_data.setdefault('relationships', []).extend(new_links)

    # Add new embeddings if not already present
    if 'x' in inference_data and 'x' in source_data:
        source_data['x'].extend(inference_data['x'])
    elif 'x' in inference_data:
        source_data['x'] = inference_data['x']
    
    return source_data

def main():
    # Load the necessary JSON files
    with open('C:/Users/pc/Desktop/ds5/Projet_AI_Cognition/output_inference/project_desc_graph+embeddings.json', 'r') as f:
        inference_data = json.load(f)
    
    with open('C:/Users/pc/Desktop/ds5/Projet_AI_Cognition/output_Graph/final_graph_data.json', 'r') as f:
        source_data = json.load(f)

    # Load the model
    model = load_model("C:/Users/pc/Desktop/ds5/Projet_AI_Cognition/Saved_Model/model.pth")  # Replace with the path to your model file

    # Perform classification and update inference data with risk criticality
    inference_data = perform_classification_and_update(inference_data, model)

    # Perform link prediction and get suggested new links
    suggested_new_links = perform_link_prediction(inference_data, source_data, model)

    # Merge the source and inference data with the suggested new links
    final_data = merge_data(source_data, inference_data, suggested_new_links)

    # Save the merged final data back to the JSON file
    with open('C:/Users/pc/Desktop/ds5/Projet_AI_Cognition/data_augmentation/final_data_augmented.json', 'w') as f:
        json.dump(final_data, f, indent=4)

    print("Updated JSON file saved as 'Projet_AI_Cognition/data_augmentation/final_data_augmented.json'")

if __name__ == "__main__":
    main()
