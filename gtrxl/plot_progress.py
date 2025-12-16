import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def main():
    stats_path = "output/gtrxl_training_stats.npy"
    if not os.path.exists(stats_path):
        print("Stats file not found.")
        return

    history = np.load(stats_path, allow_pickle=True).item()
    epochs = range(1, len(history['train_loss']) + 1)

    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)

    c_train = '#1f77b4'      # Blue
    c_test = '#d62728'       # Red for contrast vs validation
    c_base = '#2ca02c'       # Green for baselines
    
    def plot_task(ax, key_train, key_test, title, y_label, baseline=None, base_label=None):
        ax.plot(epochs, history[key_train], color=c_train, linewidth=2.5, label='Train')
        ax.plot(epochs, history[key_test], color=c_test, linewidth=2.5, label='Validation')
        
        if baseline is not None:
             ax.axhline(y=baseline, color=c_base, linestyle='--', linewidth=1.5, alpha=0.8, label=base_label or 'Baseline')

        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        data_min = min(min(history[key_train]), min(history[key_test]))
        data_max = max(max(history[key_train]), max(history[key_test]))
        
        view_min = data_min
        view_max = data_max
        if baseline:
             if baseline < view_min and (view_min - baseline) < (data_max - data_min) * 2:
                 view_min = baseline
             if baseline > view_max and (baseline - view_max) < (data_max - data_min) * 2:
                 view_max = baseline
                 
        padding = (view_max - view_min) * 0.15
        ax.set_ylim(view_min - padding, view_max + padding)
        
        ax.legend(frameon=True, fontsize=10, loc='best')

    plot_task(axes[0], 'train_loss', 'test_loss', "Total Loss", "Loss Value")
    
    plot_task(axes[1], 'train_next', 'test_next', "Next State Loss", "Cross Entropy Loss", 
              baseline=1.38, base_label='Random Neighbor (1.38)')

    plot_task(axes[2], 'train_future', 'test_future', "Future Occupancy Loss", "BCE Loss",
              baseline=0.25, base_label='Sparsity Ref (~0.25)')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    save_path = "output/presentation_loss_curves.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved graph to {save_path}")

if __name__ == "__main__":
    main()