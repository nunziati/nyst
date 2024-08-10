import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class CustomDataset(Dataset):
    """
    Expected npy format:
    {
        'signals': np.array['left_position X', 'left_position Y', 
                    'right_position X', 'right_position Y', 'left_speed X', 'left_speed Y', 
                    'right_speed X', 'right_speed Y']
        'resolutions': 'val_resolution'
        'patients': np.array: patient folder number, 
        'samples': np.array: patient video, clip and number of valid resolutions for this clip # (n_samples, n_clip, valid_resolution)
        'labels': np.array, # (len(samples)*len(val_resolution), 1)
    }
    """

    def __init__(self, csv_input_file, csv_label_file, preprocess=None, transform=None):
        # Load the CSV file
        self.input_data = pd.read_csv(csv_input_file)
        self.label_data = pd.read_csv(csv_label_file)

        # Perform the join on the 'video' column
        self.merged_data = pd.merge(self.input_data, self.label_data, on='video', how='left')
        print(self.merged_data.head(6))

        # Applica la funzione di preprocessing se fornita
        if preprocess:
            data = preprocess(self.merged_data)
        else:
            data = self.merged_data

        # Data manipulation
        data = self.exctraction_values(data)

        # Extract the different components of the dataset
        self.signals = data['signals']
        self.resolutions = data['resolutions']
        self.patients = data['patients']
        self.samples = data['samples']
        self.labels = data['labels']

        # Store the transformation function (if any)
        self.transform = transform

    def exctraction_values(self, merged_data):
        """
        Preprocessing di default dei dati uniti. Qui viene implementata la logica di base
        per estrarre i segnali, le risoluzioni, i pazienti, i campioni e le etichette.
        """
        # Estrai i segnali
        signals = merged_data[['left_position X', 'left_position Y', 
                                'right_position X', 'right_position Y',
                                'left_speed X', 'left_speed Y', 
                                'right_speed X', 'right_speed Y']].to_numpy()

        # Estrai le risoluzioni
        resolutions = merged_data['resolutions'].to_numpy().reshape(-1, 1)

        # Estrai le informazioni sui pazienti (questo può dipendere dai tuoi dati)
        patients = merged_data['video'].apply(lambda x: x.split('\\')[-1].split('_')[0]).unique().to_numpy().reshape(-1, 1)


        # Estrarre campioni e informazioni sui video
        samples = merged_data.apply(
            lambda x: [
                x['video'].split('_')[1],  # '001', '002', '001'
                x['video'].split('_')[2].split('.')[0],  # '001', '001', '002'
                x['resolutions']  # 1 se la risoluzione è valida
            ], 
            axis=1
        ).to_numpy()

        # Estrai le etichette
        labels = merged_data['label'].to_numpy().reshape(-1, 1)

        return {
            'signals': signals,
            'resolutions': resolutions,
            'patients': patients,
            'samples': samples,
            'labels': labels
        }

    # Return the number of samples in the dataset
    def __len__(self):
        return len(self.samples)

    # Return the signal and label
    def __getitem__(self, idx):
        # Convert tensor index to list if necessary
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Retrieve the sample and its corresponding label
        signal = self.signals[idx]
        label = self.labels[idx]

        # Apply the transformation to the sample if provided
        if self.transform:
            sample = self.transform(sample)

        return signal, label


if __name__ == '__main__':
    # Replace with the actual paths to your CSV files
    csv_input_file = 'path_to_your_input_file.csv'
    csv_label_file = 'path_to_your_label_file.csv'

    # Create an instance of the CustomDataset to trigger the print
    dataset = CustomDataset(csv_input_file, csv_label_file)