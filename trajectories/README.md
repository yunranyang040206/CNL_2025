# Long Term Dependency Trajectories

This directory contains scripts for generating and visualizing "Ground Truth" data to test Inverse Reinforcement Learning (specifically SWIRL).

Instead of training an imperfect agent, we use a **Rule-Based Oracle** to generate deterministic "Expert" trajectories that switch between two distinct latent modes based on internal states (Thirst).

## 1. Data Generation Rules

The data simulates an animal subject to homeostatic regulation (Thirst) in a 5x5 GridWorld.

### Environment

- **Grid Size**: 5x5 (States 0-24)
- **Actions**: 5 (Up, Down, Left, Right, Stay)
- **Horizon**: 500 steps per episode
- **Episodes**: 200

### Behavioral Logic (The State Machine)

The agent switches between two latent modes ($z$) based on the following rules:

1. **Start**: Agent initializes in **Mode 1 (Thirsty)**.
2. **Mode 1 (Water Seeking)**:
    - **Policy**: Optimal navigation. The agent navigates directly to the "Start Node" (20), then executes a strict path along the bottom row (`20->21->22->23->24`).
    - **Transition**: Upon reaching Node 24, the agent drinks, becomes **Satiated**, and switches to Mode 0.
3. **Mode 0 (Exploring)**:
    - **Policy**: Random Brownian motion (uniform probability over valid actions).
    - **Transition**: The agent remains Satiated for exactly **100 timesteps**. After this timer expires, it becomes **Thirsty** and switches back to Mode 1.

## 2. Generated Artifacts

Running the generation script produces the following files in `output/`:

| File | Shape | Description |
| :--- | :--- | :--- |
| `xs.npy` | `(N, T+1)` | State trajectories (0-24 flat indices). |
| `acs.npy` | `(N, T)` | Actions taken. |
| `zs.npy` | `(N, T)` | **Ground Truth Latent Modes** (0=Explore, 1=Water). |
| `RG.npy` | `(2, 25, 25)` | **Ground Truth Reward Tensor** defining the sparse reward logic. |
| `trans_prob.npy` | `(25, 5, 25)` | Deterministic transition dynamics of the grid. |

---

## 3. Visualization

The plotting suite generates a dashboard to verify the data integrity.

```bash
python plot_artifacts.py
