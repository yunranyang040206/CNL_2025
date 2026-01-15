import torch
import torch.nn as nn
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import sys
import os

# Add parent directory to path to import utils
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from utils.dataset import TrajectoryDataset

class MarkovianMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 5)
        )
    def forward(self, x):
        return self.net(x[:, -1, :]) # Only uses the current state (last in history window)

def train_baseline():
    print("Training Markovian MLP Baseline...")
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(script_dir, '../data/animal_data_train.csv')
    val_path = os.path.join(script_dir, '../data/animal_data_val.csv')
    
    train_dataset = TrajectoryDataset(train_path)
    val_dataset = TrajectoryDataset(val_path)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    model = MarkovianMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    metrics = {'acc': [], 'val_acc': []}
    for epoch in range(100):
        # Train
        model.train()
        correct, total = 0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            output = model(batch['history'])
            loss = criterion(output, batch['target_a'])
            loss.backward(); optimizer.step()
            _, pred = torch.max(output, 1)
            total += batch['target_a'].size(0)
            correct += (pred == batch['target_a']).sum().item()
        acc = 100 * correct / total
        metrics['acc'].append(acc)
        
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
        
        print(f"Epoch {epoch+1:02d} | Train Acc: {acc:.2f}% | Val Acc: {val_acc:.2f}%")

    import json
    
    results_dir = os.path.join(script_dir, '../results')
    os.makedirs(results_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(results_dir, 'mlp_weights.pth'))
    
    with open(os.path.join(results_dir, 'mlp_stats.json'), 'w') as f:
        json.dump(metrics, f)
        
    print("Baseline Weights and Stats saved to results/")

if __name__ == "__main__": train_baseline()