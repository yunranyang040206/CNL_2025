import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def main():
    stats_path = "output/swirl_stats.npz"
    save_path = "output/swirl_training_results.png"

    if not os.path.exists(stats_path):
        print(f"Stats file not found at {stats_path}. Please run training first.")
        return

    # Load Data
    data = np.load(stats_path)
    t_ll = data['t_ll']
    v_ll = data['v_ll']
    mode_acc = data['mode_acc']
    
    iterations = np.arange(1, len(t_ll) + 1)

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    # Changed to 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    # 1. Log Likelihood
    ax = axes[0]
    ax.plot(iterations, t_ll, label='Train LL', color='#1f77b4', linewidth=2.5, marker='o', markersize=4)
    ax.plot(iterations, v_ll, label='Val LL', color='#d62728', linewidth=2.5, marker='s', markersize=4)
    
    # Baselines
    ax.axhline(y=-1.40, color='#2ca02c', linestyle='--', linewidth=2, label='SWIRL Baseline (-1.40)')
    ax.axhline(y=-1.65, color='#7f7f7f', linestyle=':', linewidth=2, label='MaxEnt IRL (-1.65)')
    
    ax.set_title("Log-Likelihood", fontsize=14, pad=10)
    ax.set_xlabel("EM Iteration", fontsize=12)
    ax.set_ylabel("Log Prob per Step", fontsize=12)
    
    # Scale setting
    ax.set_ylim(bottom=-1.7)
    
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)

    # 2. Mode Accuracy
    ax = axes[1]
    if np.max(mode_acc) > 0:
        ax.plot(iterations, mode_acc * 100, color='#9467bd', linewidth=2.5, marker='d')
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        final_acc = mode_acc[-1] * 100
        ax.set_title(f"Mode Segmentation Accuracy (Final: {final_acc:.1f}%)", fontsize=14, pad=10)
    else:
        ax.text(0.5, 0.5, "No Ground Truth Available", ha='center', va='center', fontsize=12, color='gray')
        ax.set_title("Mode Accuracy", fontsize=14, pad=10)
    
    ax.set_xlabel("EM Iteration", fontsize=12)
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Saved training plots to {save_path}")

if __name__ == "__main__":
    main()