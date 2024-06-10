import os
import torch
from torch.utils.data import Dataset
import numpy as np

class CustomDataset(Dataset):
    """
    Expected npy format:
    {
        'signals': ['left_position_x', 'left_position_y', ...], # (n_signals,);
        'patients': np.array, # (n_samples, 1) 
        'samples': np.array, # (n_samples, n_signals, n_timesteps)
        'labels': np.array, # (n_samples, 1)
    }
    """

    def __init__(self, npy_file, preprocess, transform=None):
        data = np.load(npy_file, allow_pickle=True)

        data = preprocess(data)

        self.signals = data['signals']
        self.patients = data['patients']
        self.samples = data['samples']
        self.labels = data['labels']

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.samples[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label

