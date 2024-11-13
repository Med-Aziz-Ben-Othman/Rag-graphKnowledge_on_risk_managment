import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .model import EnhancedSAGEModel
from data_loading_processing import load_saved_data

def train_model(input_dim, hidden_dim=1024, output_dim=3, learning_rate=0.0001, weight_decay=3e-3, max_epochs=100):
    # Load processed data
    data = load_saved_data()
    
    # Initialize model, optimizer, and scheduler
    model = EnhancedSAGEModel(input_dim, hidden_dim, output_dim)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Split data into train and validation masks
    num_nodes = data.y.size(0)
    train_mask = torch.rand(num_nodes) < 0.8
    val_mask = ~train_mask

    # Training loop
    patience = 3
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        node_logits = model(data.x, data.edge_index)
        train_loss = loss_fn(node_logits[train_mask], data.y[train_mask])
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        # Validation
        with torch.no_grad():
            model.eval()
            val_loss = loss_fn(node_logits[val_mask], data.y[val_mask])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} with best validation loss: {best_val_loss:.4f}")
                break
        scheduler.step(val_loss)
        print(f"Epoch {epoch}, Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

    return model
