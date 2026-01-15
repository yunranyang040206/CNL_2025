import numpy as np
import pandas as pd
import os

def generate_thirsty_start_data(num_trajs=100, steps=100, output_file='animal_data.csv'):
    data = []
    WATER_POS = (0, 0)
    TREE_POS = (4, 4)
    actions_map = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}
    
    for traj_id in range(num_trajs):
        curr_pos = [2, 2] # Start in center
        
        # --- MODIFICATION START ---
        thirst = 10   # Start at threshold so the first action is 'Go to Water'
        fatigue = 0   # Start fresh for fatigue
        # --- MODIFICATION END ---
        
        for t in range(steps):
            # Decision Logic
            if thirst >= 20 and thirst >= fatigue:
                target = WATER_POS
            elif fatigue >= 20 and fatigue > thirst:
                target = TREE_POS
            else:
                target = None

            # Action Selection (Shortest path to target)
            if target:
                dr, dc = target[0] - curr_pos[0], target[1] - curr_pos[1]
                if abs(dr) >= abs(dc) and dr != 0:
                    action_idx = 0 if dr < 0 else 1
                elif dc != 0:
                    action_idx = 2 if dc < 0 else 3
                else:
                    action_idx = 4
            else:
                action_idx = np.random.choice([0, 1, 2, 3, 4])

            # Apply Move
            move = actions_map[action_idx]
            new_r = np.clip(curr_pos[0] + move[0], 0, 4)
            new_c = np.clip(curr_pos[1] + move[1], 0, 4)
            
            data.append({
                'traj_id': traj_id, 'step': t, 
                'row': curr_pos[0], 'col': curr_pos[1], 
                'action': action_idx
            })
            
            # Update Latent States
            thirst += 1
            fatigue += 1
            if (new_r, new_c) == WATER_POS: thirst = 0
            if (new_r, new_c) == TREE_POS: fatigue = 0
            curr_pos = [new_r, new_c]

    df = pd.DataFrame(data)
    
    # Split 80/20
    traj_ids = df['traj_id'].unique()
    np.random.shuffle(traj_ids)
    split_idx = int(0.8 * len(traj_ids))
    train_ids = traj_ids[:split_idx]
    val_ids = traj_ids[split_idx:]
    
    train_df = df[df['traj_id'].isin(train_ids)]
    val_df = df[df['traj_id'].isin(val_ids)]
    
    train_file = output_file.replace('.csv', '_train.csv')
    val_file = output_file.replace('.csv', '_val.csv')
    
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    
    print(f"Generated {len(train_ids)} train trajectories -> {train_file}")
    print(f"Generated {len(val_ids)} val trajectories -> {val_file}")

    return train_df, val_df

if __name__ == "__main__":
    generate_thirsty_start_data()