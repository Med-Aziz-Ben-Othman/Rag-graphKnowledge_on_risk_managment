# main.py

import torch
from data_loading_processing import load_saved_data  # Custom data loading function
from Modeling.train import train_model
from Modeling.config import config

# Load preprocessed data
data = load_saved_data()

# Train model with specified config parameters
model = train_model(
    input_dim=config['input_dim'],
    hidden_dim=config['hidden_dim'],
    output_dim=config['output_dim'],
    learning_rate=config['learning_rate'],
    weight_decay=config['weight_decay'],
    max_epochs=config['max_epochs']
)

# Save the trained model to a .pth file
model_save_path = 'Projet_AI_Cognition/Saved_Model/model.pth'
torch.save(model.state_dict(), model_save_path)

print(f"Training complete. Model saved to {model_save_path}.")
