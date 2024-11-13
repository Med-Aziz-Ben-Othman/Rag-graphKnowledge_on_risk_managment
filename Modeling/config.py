# Modeling/config.py

config = {
    'input_dim': 1536,   # Set this dynamically based on data
    'hidden_dim': 1024,
    'output_dim': 3,    # Number of classes (Low, Medium, High)
    'ff_hidden_dim': 128,
    'dropout_rate': 0.5,
    'learning_rate': 0.0001,
    'weight_decay': 3e-3,
    'max_epochs': 100,
    'patience': 3,
    'train_split': 0.8
}
