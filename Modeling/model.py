# Modeling/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class EnhancedSAGEModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, ff_hidden_dim=128, dropout_rate=0.5):
        super(EnhancedSAGEModel, self).__init__()
        
        self.sage1 = SAGEConv(input_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)
        self.sage3 = SAGEConv(hidden_dim, hidden_dim)
        
        self.ff_layer1 = nn.Linear(hidden_dim, ff_hidden_dim)
        self.ff_layer2 = nn.Linear(ff_hidden_dim, hidden_dim)
        
        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.link_predictor = nn.Linear(hidden_dim * 2, 1)
        
        self.residual1 = nn.Linear(input_dim, hidden_dim)
        self.residual2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, edge_index):
        residual = self.residual1(x)
        x = self.sage1(x, edge_index)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x += residual

        residual = self.residual2(x)
        x = self.sage2(x, edge_index)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x += residual

        x = self.sage3(x, edge_index)
        x = F.relu(self.ff_layer1(x))
        x = self.dropout(x)
        x = self.ff_layer2(x)
        
        node_logits = self.classifier(x)
        self.node_embeddings = x
        return node_logits

    def predict_links(self, node_pair_indices):
        if len(node_pair_indices[0]) == 0:
            return torch.tensor([])

        node1_embeddings = self.node_embeddings[node_pair_indices[0]]
        node2_embeddings = self.node_embeddings[node_pair_indices[1]]

        node1_embeddings = F.normalize(node1_embeddings, p=2, dim=1)
        node2_embeddings = F.normalize(node2_embeddings, p=2, dim=1)
        
        concatenated_embeddings = torch.cat([node1_embeddings, node2_embeddings], dim=1)
        link_logits = self.link_predictor(concatenated_embeddings)

        return torch.sigmoid(link_logits)
