import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg') # WSL Compatibility
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

import sys
import os

# Add parent directory to path to import utils
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from utils.dataset import TrajectoryDataset

class Transformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, max_window=30):
        super().__init__()
        # FIX 1: Use discrete embeddings for the 25 grid cells
        self.state_embed = nn.Embedding(25, d_model)
        
        # FIX 2: Add Positional Encodings so the model understands "Time"
        self.pos_embed = nn.Embedding(max_window, d_model)
        
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=3)
        self.reward_head = nn.Linear(d_model, 5)

    def forward(self, x):
        # x is currently (Batch, Window, 2) coords
        # Convert (r, c) to a single index 0-24
        state_indices = (x[:, :, 0] * 5 + x[:, :, 1]).long()
        
        # Apply State + Positional Embeddings
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = self.state_embed(state_indices) + self.pos_embed(positions)
        
        # Causal Mask
        mask = torch.triu(torch.ones(x.size(1), x.size(1), device=x.device) == 1, diagonal=1)
        
        latent = self.transformer(x, mask=mask)
        
        # Predict based on the "Current" latent state (the last step in history)
        return self.reward_head(latent[:, -1, :])

def train():
    print("Training Transformer...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(script_dir, '../data/animal_data_train.csv')
    val_path = os.path.join(script_dir, '../data/animal_data_val.csv')
    
    train_dataset = TrajectoryDataset(train_path)
    val_dataset = TrajectoryDataset(val_path)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    model = Transformer()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    metrics = {'acc': [], 'loss': [], 'val_acc': []}
    for epoch in range(20):
        # Training
        model.train()
        correct, total, epoch_loss = 0, 0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            output = model(batch['history'])
            loss = criterion(output, batch['target_a'])

            loss.backward(); optimizer.step()
            epoch_loss += loss.item()
            _, pred = torch.max(output, 1)
            total += batch['target_a'].size(0)
            correct += (pred == batch['target_a']).sum().item()
        
        train_acc = 100 * correct / total
        metrics['acc'].append(train_acc)
        metrics['loss'].append(epoch_loss / len(train_loader))
        
        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                output = model(batch['history'])
                _, pred = torch.max(output, 1)
                val_total += batch['target_a'].size(0)
                val_correct += (pred == batch['target_a']).sum().item()
        
        val_acc = 100 * val_correct / val_total
        metrics['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1:02d} | Loss: {metrics['loss'][-1]:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    import json
    
    results_dir = os.path.join(script_dir, '../results')
    os.makedirs(results_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(results_dir, 'transformer_weights.pth'))
    
    with open(os.path.join(results_dir, 'transformer_stats.json'), 'w') as f:
        json.dump(metrics, f)
        
    print("Weights and Stats saved to results/")

if __name__ == "__main__": train()