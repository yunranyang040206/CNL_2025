import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

GRID_N = 5

def idx_to_rc(i):
    return divmod(i, GRID_N)

def plot_trajectories(xs, output_path, num_trajectories=10, timesteps=None):
    """
    Plots trajectories on a 2D grid.
    xs: (N_EPISODES, T_STEPS) array of state indices
    output_path: path to save the plot
    num_trajectories: number of trajectories to plot
    timesteps: number of timesteps to plot (optional)
    """
    plt.figure(figsize=(8, 8))
    
    # Plot grid
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xlim(-0.5, GRID_N - 0.5)
    plt.ylim(GRID_N - 0.5, -0.5) # Invert y-axis to match matrix coordinates (0,0 at top-left)
    plt.xticks(range(GRID_N))
    plt.yticks(range(GRID_N))
    
    # Slice xs if timesteps is specified
    if timesteps is not None:
        xs = xs[:, :timesteps]

    # Plot trajectories
    n_plot = min(num_trajectories, xs.shape[0])
    colors = plt.cm.jet(np.linspace(0, 1, n_plot))
    
    for i in range(n_plot):
        traj = xs[i]
        rows, cols = zip(*[idx_to_rc(s) for s in traj])
        
        # Add jitter to see overlapping paths
        jitter = 0.1
        rows = np.array(rows) + np.random.uniform(-jitter, jitter, size=len(rows))
        cols = np.array(cols) + np.random.uniform(-jitter, jitter, size=len(cols))
        
        plt.plot(cols, rows, '-', color=colors[i], alpha=0.6, linewidth=1)
        plt.plot(cols[0], rows[0], 'go', markersize=5) # Start
        plt.plot(cols[-1], rows[-1], 'rx', markersize=5) # End

    total_episodes = xs.shape[0]
    current_timesteps = xs.shape[1]
    plt.title(f'First {n_plot} of {total_episodes} Trajectories\n(Duration: {current_timesteps} steps per episode)')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot trajectories')
    parser.add_argument('xs_file', type=str, nargs='?', default="data/xs.npy", help='Path to xs.npy file')
    parser.add_argument('--num_trajectories', type=int, default=10, help='Number of trajectories to plot')
    parser.add_argument('--timesteps', type=int, default=None, help='Number of timesteps to plot per trajectory')
    args = parser.parse_args()

    xs_path = args.xs_file
    
    if not os.path.exists(xs_path):
        print(f"Error: {xs_path} not found.")
        exit(1)

    print(f"Loading data from {xs_path}...")
    xs = np.load(xs_path)
    
    # Save to same directory as input file with timestamp and num_trajectories
    from datetime import datetime
    output_dir = os.path.dirname(xs_path)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f'trajectories_n{args.num_trajectories}_{timestamp}.png'
    output_path = os.path.join(output_dir, output_filename)
    
    plot_trajectories(xs, output_path, args.num_trajectories, args.timesteps)
