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
        # Load the data from the npy file
        data = np.load(npy_file, allow_pickle=True)

        # Apply preprocessing if provided
        data = preprocess(data)

        # Extract the different components of the dataset
        self.signals = data['signals']
        self.patients = data['patients']
        self.samples = data['samples']
        self.labels = data['labels']

        # Store the transformation function (if any)
        self.transform = transform

    # Return the number of samples in the dataset
    def __len__(self):
        return len(self.samples)

    # Return the sample and label
    def __getitem__(self, idx):
        # Convert tensor index to list if necessary
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Retrieve the sample and its corresponding label
        sample = self.samples[idx]
        label = self.labels[idx]

        # Apply the transformation to the sample if provided
        if self.transform:
            sample = self.transform(sample)

        return sample, label

