import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 1. Setup paths and import your Transformer architecture
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from transformer import Transformer 

def generate_at_node_graph():
    results_dir = os.path.join(script_dir, '../results')
    model_path = os.path.join(results_dir, 'transformer_weights.pth')
    
    model = Transformer()
    if not os.path.exists(model_path):
        print(f"Error: Could not find weights at {model_path}")
        return
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # 2. Define Contexts and Probes at the SAME location
    # Case A: At the Tree (4,4). Just finished "Resting". Should want to leave for Water.
    tree_pos = torch.tensor([[[4.0, 4.0]]])
    thirsty_history = torch.full((1, 24, 2), 4.0) 
    
    # Case B: At the Water (0,0). Just finished "Drinking". Should want to leave for Tree.
    water_pos = torch.tensor([[[0.0, 0.0]]])
    tired_history = torch.full((1, 24, 2), 0.0)

    action_names = ["UP (to Water)", "DOWN (to Home)", "LEFT (to Water)", "RIGHT (to Home)", "STAY"]
    
    with torch.no_grad():
        # Get Probabilities for being at the Tree
        logits_tree = model(torch.cat([thirsty_history, tree_pos], dim=1))
        probs_tree = torch.softmax(logits_tree, dim=1).squeeze().numpy()

        # Get Probabilities for being at the Water
        logits_water = model(torch.cat([tired_history, water_pos], dim=1))
        probs_water = torch.softmax(logits_water, dim=1).squeeze().numpy()

    # 3. Plotting
    x = np.arange(len(action_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - width/2, probs_tree, width, label='At Home (4,4) after 25 steps', color='#e67e22', alpha=0.8)
    ax.bar(x + width/2, probs_water, width, label='At Water (0,0) after 25 steps', color='#3498db', alpha=0.8)

    ax.set_ylabel('Probability')
    ax.set_title('Transformer Decision Probe: Behavior AT the Resource Nodes')
    ax.set_xticks(x)
    ax.set_xticklabels(action_names)
    ax.legend()
    ax.set_ylim(0, 1.1)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    save_path = os.path.join(results_dir, 'at_node_comparison_graph.png')
    plt.savefig(save_path)
    print(f"Graph saved to {save_path}")

if __name__ == "__main__":
    generate_at_node_graph()