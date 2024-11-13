# Modeling/utils.py

import torch

def split_data(y, train_split=0.8):
    train_mask = torch.rand(y.size(0)) < train_split
    val_mask = ~train_mask
    return train_mask, val_mask
def make_all_lowercase(data):
    # Recursively make all keys lowercase in the JSON structure
    if isinstance(data, dict):
        return {k.lower(): make_all_lowercase(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [make_all_lowercase(item) for item in data]
    elif isinstance(data, str):
        return data.lower()
    else:
        return data
