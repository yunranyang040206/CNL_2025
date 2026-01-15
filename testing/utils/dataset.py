import torch
import pandas as pd
from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    def __init__(self, csv_file, seq_len=25):
        self.df = pd.read_csv(csv_file)
        self.samples = []
        for _, group in self.df.groupby('traj_id'):
            states = group[['row', 'col']].values / 4.0
            actions = group['action'].values
            for i in range(len(states) - seq_len):
                self.samples.append({
                    'history': torch.tensor(states[i : i + seq_len], dtype=torch.float32),
                    'target_a': torch.tensor(actions[i + seq_len], dtype=torch.long)
                })
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]
