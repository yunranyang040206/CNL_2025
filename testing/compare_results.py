import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_stats(filepath, key='acc'):
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found.")
        return []
    with open(filepath, 'r') as f:
        data = json.load(f)
        return data.get(key, data.get('acc', []))

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    
    mlp_acc = load_stats(os.path.join(results_dir, 'mlp_stats.json'), 'acc')
    transformer_acc = load_stats(os.path.join(results_dir, 'transformer_stats.json'), 'acc')
    swirl_acc = load_stats(os.path.join(results_dir, 'swirl_s2_stats.json'), 'acc')
    
    # Truncate to minimum length for consistency
    min_len = min(len(mlp_acc), len(transformer_acc), len(swirl_acc)) if (mlp_acc and transformer_acc and swirl_acc) else 0
    
    if min_len > 0:
        mlp_acc = mlp_acc[:min_len]
        transformer_acc = transformer_acc[:min_len]
        swirl_acc = swirl_acc[:min_len]
        
    plt.figure(figsize=(10, 6))
    
    if mlp_acc:
        plt.plot(range(1, len(mlp_acc) + 1), mlp_acc, label='Markovian MLP', marker='o')
    if transformer_acc:
        plt.plot(range(1, len(transformer_acc) + 1), transformer_acc, label='Transformer', marker='s')
    if swirl_acc:
        plt.plot(range(1, len(swirl_acc) + 1), swirl_acc, label='Swirl S-2', marker='^')
        
    import matplotlib.ticker as ticker
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(2))
    
    plt.xlabel('Epoch / Iteration')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Comparison - Accuracy')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(results_dir, 'model_comparison.png')
    plt.savefig(save_path)
    print(f"Comparison plot saved to {save_path}")

if __name__ == "__main__":
    main()
