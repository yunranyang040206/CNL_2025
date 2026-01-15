import pandas as pd
import numpy as np
import matplotlib
# Force non-interactive backend for WSL/Terminal environments
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_animal_data(csv_file='animal_data_train.csv'):
    # 1. Load the data
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Please run the generator first.")
        return
    
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} steps from {csv_file}")

    # 2. Create an Occupancy Heatmap (Overall Behavior)
    # This shows if the animal is actually visiting the Water (0,0) and Tree (4,4)
    occupancy = np.zeros((5, 5))
    for _, row in df.iterrows():
        occupancy[int(row['row']), int(row['col'])] += 1

    plt.figure(figsize=(8, 6))
    sns.heatmap(occupancy, annot=True, fmt=".0f", cmap="YlOrRd")
    
    # Annotate landmarks on the heatmap
    plt.text(0.5, 0.5, 'WATER', ha='center', va='center', color='blue', weight='bold')
    plt.text(4.5, 4.5, 'REST', ha='center', va='center', color='green', weight='bold')
    
    plt.title("5x5 Gridworld: Overall Occupancy")
    plt.xlabel("Column (col)")
    plt.ylabel("Row (row)")
    plt.savefig('grid_occupancy_heatmap.png')
    print("Saved heatmap to 'grid_occupancy_heatmap.png'")

    # 3. Trace a Single Trajectory (Temporal Behavior)
    # We'll pick traj_id 0 to see the sequence of moves
    plt.figure(figsize=(6, 6))
    single_traj = df[df['traj_id'] == 49]
    
    # Add a small amount of jitter so overlapping paths are visible
    jitter_r = single_traj['row'] + np.random.normal(0, 0.05, len(single_traj))
    jitter_c = single_traj['col'] + np.random.normal(0, 0.05, len(single_traj))
    
    plt.plot(jitter_c, jitter_r, color='gray', alpha=0.5, linewidth=1)
    scatter = plt.scatter(jitter_c, jitter_r, c=single_traj['step'], cmap='viridis', s=20, zorder=3)
    
    # Plot Landmarks
    plt.scatter(0, 0, color='blue', marker='s', s=200, label='Water (0,0)')
    plt.scatter(4, 4, color='green', marker='^', s=200, label='Rest (4,4)')
    
    plt.colorbar(scatter, label='Time Step')
    plt.gca().invert_yaxis() # Match grid indexing (row 0 at top)
    plt.title("Single Trajectory Path (Traj ID: 49)")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.savefig('single_trajectory_trace.png')
    print("Saved trajectory trace to 'single_trajectory_trace.png'")

if __name__ == "__main__":
    import os
    visualize_animal_data()