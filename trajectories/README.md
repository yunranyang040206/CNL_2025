# Trajectory Generation

Generate and visualize agent trajectories in a gridworld environment.

## Quick Start

### Generate Trajectories

```bash
# Use default config
python generate_trajectories.py

# Use custom config
python generate_trajectories.py --config configs/default.yaml
```

**Output:** Creates a timestamped folder in `output/` containing:

- `xs.npy` - State trajectories
- `acs.npy` - Action trajectories  
- `zs.npy` - Mode trajectories
- `RG.npy` - Reward grids
- `trans_probs.npy` - Transition probabilities
- `config.yaml` - Copy of configuration used

### Plot Trajectories

```bash
# Basic usage (plots first 10 trajectories)
python plot_trajectories.py output/default_20231124_143000/xs.npy

# Plot specific number of trajectories
python plot_trajectories.py output/default_20231124_143000/xs.npy --num_trajectories 5

# Plot only first N timesteps
python plot_trajectories.py output/default_20231124_143000/xs.npy --timesteps 100

# Combine options
python plot_trajectories.py output/default_20231124_143000/xs.npy --num_trajectories 3 --timesteps 50
```

**Output:** Saves `trajectories_n{num}_YYYYMMDD_HHMMSS.png` in the same directory as the input file (e.g., `trajectories_n10_20231124_143000.png`).

## Configuration

See example configs in `configs/` folder. Key parameters:

- **Grid size:** `grid.size` (e.g., 5 for 5×5 grid)
- **Episodes:** `trajectories.n_episodes`
- **Steps per episode:** `trajectories.t_steps`
- **Rewards:** Define in `rewards.state_based` and `rewards.sequence_based`

For detailed configuration options, see the config files in `configs/`.
