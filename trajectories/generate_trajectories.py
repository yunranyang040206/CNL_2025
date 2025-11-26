import numpy as np
import os

# ===========================
# CONFIGURATION
# ===========================
CONFIG = {
    "grid_size": 5,
    "n_episodes": 1,
    "max_steps": 500,
    "explore_duration": 100,
    "seq_start_node": 20,
    "seq_end_node": 24,
    "noise": 0.05,
    "output_dir": "./output",
}

N_STATES = CONFIG["grid_size"] ** 2
N_ACTIONS = 5
ACTION_MAP = {0: ((-1, 0)), 1: ((1, 0)), 2: ((0, -1)), 3: ((0, 1)), 4: ((0, 0))}


def get_next_state(state, action):
    r, c = divmod(state, CONFIG["grid_size"])
    dr, dc = ACTION_MAP[action]
    r2 = np.clip(r + dr, 0, CONFIG["grid_size"] - 1)
    c2 = np.clip(c + dc, 0, CONFIG["grid_size"] - 1)
    return r2 * CONFIG["grid_size"] + c2


# ===========================
# STATIC FILES
# ===========================
def generate_static_files():
    # 1. Reward Grid: (2 Modes, 25 From_State, 25 To_State)
    RG = np.zeros((2, N_STATES, N_STATES), dtype=np.float32)
    RG[0, :, :] = 0.0
    RG[1, 23, 24] = 10.0  # Sparse Reward only in Water Mode

    # 2. Transition Probability Matrix: (25 From_State, 5 Actions, 25 To_State)
    # This defines the physics of the grid
    trans_prob = np.zeros((N_STATES, N_ACTIONS, N_STATES), dtype=np.float32)

    for s in range(N_STATES):
        for a in range(N_ACTIONS):
            next_s = get_next_state(s, a)
            # Since grid is deterministic, probability is 1.0 for the outcome
            trans_prob[s, a, next_s] = 1.0

    return RG, trans_prob


# ===========================
# AGENT LOGIC
# ===========================
def get_water_action(curr_state, has_touched_start):
    grid_n = CONFIG["grid_size"]
    curr_r, curr_c = divmod(curr_state, grid_n)
    if has_touched_start:
        if curr_r == 4:
            return 3  # Right
        return 1  # Down
    target_r, target_c = divmod(CONFIG["seq_start_node"], grid_n)
    if curr_r < target_r:
        return 1
    if curr_c > target_c:
        return 2
    return 4


def generate_trajectories():
    all_xs, all_acs, all_zs = [], [], []

    for i in range(CONFIG["n_episodes"]):
        curr = np.random.randint(0, N_STATES)
        is_thirsty = True
        thirst_timer = 0
        has_touched_start = curr == CONFIG["seq_start_node"]

        traj_x = [curr]
        traj_a = []
        traj_z = []

        for t in range(CONFIG["max_steps"]):
            # --- DECISION ---
            if is_thirsty:
                z = 1
                if curr == CONFIG["seq_start_node"]:
                    has_touched_start = True
                action = get_water_action(curr, has_touched_start)
            else:
                z = 0
                action = np.random.randint(0, 4)

            if np.random.rand() < CONFIG["noise"]:
                action = np.random.randint(0, 4)

            # --- STEP ---
            next_s = get_next_state(curr, action)
            traj_x.append(next_s)
            traj_a.append(action)
            traj_z.append(z)
            curr = next_s

            # --- TRANSITIONS ---
            if is_thirsty:
                # Drink Water -> Satiated
                if curr == CONFIG["seq_end_node"] and has_touched_start:
                    is_thirsty = False
                    thirst_timer = 0
            else:
                # Wait -> Thirsty (Check for >= 100)
                thirst_timer += 1
                if thirst_timer >= CONFIG["explore_duration"]:
                    is_thirsty = True
                    has_touched_start = False

        all_xs.append(np.array(traj_x, dtype=np.int64))
        all_acs.append(np.array(traj_a, dtype=np.int64))
        all_zs.append(np.array(traj_z, dtype=np.int64))

    return all_xs, all_acs, all_zs


if __name__ == "__main__":
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # Generate both static files
    RG, trans_prob = generate_static_files()

    # Generate trajectories
    xs, acs, zs = generate_trajectories()

    # Save Everything
    np.save(f"{CONFIG['output_dir']}/RG.npy", RG)
    np.save(f"{CONFIG['output_dir']}/trans_prob.npy", trans_prob)

    np.save(f"{CONFIG['output_dir']}/xs.npy", np.array(xs, dtype=object))
    np.save(f"{CONFIG['output_dir']}/acs.npy", np.array(acs, dtype=object))
    np.save(f"{CONFIG['output_dir']}/zs.npy", np.array(zs, dtype=object))

    print(f"Data Generated in {CONFIG['output_dir']}")
