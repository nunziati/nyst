import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import sys
import copy

# Aggiungi la directory 'code' al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nyst.dataset.splitting_dataset import *
from nyst.dataset.preprocess_function import *




class NystDataset(Dataset):
    def __init__(self, filename, std):
        
        self.std = std

        self.data = pd.read_csv(filename)
        
        # Exctract data into a dictionary
        self.extr_data = self.exctraction_values()
        print('\n\t ---> Data extraction step COMPLETED\n')

        # METODO DA RIVEDERE, SE SERVISSE
        # Filter the invalid data             
        # self.fil_data, self.invalid_video_info = self.filtering_invalid_data()
        # print('\n\t ---> Filtering invalid data step COMPLETED\n')

        # Normalization signals            
        self.fil_norm_data = self.normalization_signals()
        print('\n\t ---> Data normalization step COMPLETED\n')

        
        
    
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

    # Extract the input/label info from the csv file
    def exctraction_values(self):
        '''
        Preprocesses the merged data to extract relevant features such as signals, resolutions,
        patient information, samples, and labels.

        Arguments:
        - merged_data (pandas.DataFrame): The dataframe containing the merged data. It is expected
        to contain columns related to positions, speeds, video information, resolutions, and labels.

        Returns:
        - dict: A dictionary containing the extracted features:
            - 'signals': A list of lists containing the extracted signal data (position and speed).
            - 'resolutions': A numpy array of resolutions from the data.
            - 'patients': A numpy array with patient IDs extracted from the video filenames.
            - 'samples': A numpy array with detailed sample information including patient ID, video number, and resolution.
            - 'labels': A numpy array of labels associated with each data entry.
        '''

        merged_data = self.data

        # Extract signaks
        signals_str = merged_data[['left_position X', 'left_position Y', 
                            'right_position X', 'right_position Y',
                            'left_speed X', 'left_speed Y', 
                            'right_speed X', 'right_speed Y']].values
        
        # Convert strings into lists of float
        signals = np.array([[parse_float_list(signal) for signal in row] for row in signals_str])

        # Resolutions extraction
        resolutions = merged_data['resolution'].to_numpy().reshape(-1)

        # Patient information extraction
        patients = merged_data['video'].apply(lambda x: x.split('_')[0].split('/')[1]) .to_numpy().reshape(-1)

        # Sample and video information extraction
        # Add new columns to the merged_data dataframe
        merged_data['patient_number'] = merged_data['video'].apply(lambda x: x.split('/')[-1].split('_')[0])
        merged_data['video_number'] = merged_data['video'].apply(lambda x: x.split('_')[1])
        merged_data['clip_number'] = merged_data['video'].apply(lambda x: x.split('_')[2].split('.')[0])

        # Select those columns and turn them into a numpy array
        samples = merged_data[['patient_number', 'video_number', 'clip_number', 'resolution']].to_numpy()

        # Label extraction
        labels = merged_data['label'].to_numpy().reshape(-1)

        return {
            'signals': signals,
            'resolutions': resolutions,
            'patients': patients,
            'samples': samples,
            'labels': labels
        }

    # Funzione aggiornata per il filtraggio dei dati
    def filtering_invalid_data(self, frames_video:int = 150, zero_threshold:float = 0.2):
        '''
        Filters out invalid data/videos based on signal dimensions and zero-speed thresholds, and also removes 
        entries associated with the same patient, video, and clip number.

        Arguments:
        - dictionary_input (dict): A dictionary containing the input data.
        - frames_video (int): The expected number of frames in each signal. Defaults to 150.
        - zero_threshold (float): The threshold for filtering out signals with excessive zero speeds. Defaults to 0.2 (20%).

        Returns:
        - tuple:
            - dict: A dictionary with filtered data, maintaining the original structure but with invalid entries removed.
            - list: A list containing information about the invalid clips that were filtered out, along with the reasons for the filtering.
        '''

        # Retrieve the input signal values
        signals = self.extr_data['signals']
        samples = self.extr_data['samples']
        valid_indices = set(range(len(signals)))  # Start with all indices being valid
        invalid_video_info = []  # To store video information and reasons for filtering
        invalid_videos = set()

        # Cycle through all signals
        for i in range(len(signals)):
            
            # Extract the specific signal values
            row = signals[i]

            # Split the signals into positions and speeds
            # positions = [parse_float_list(pos) if isinstance(pos, str) else pos for pos in row[:4]]
            speeds = [parse_float_list(speed) if isinstance(speed, str) else speed for speed in row[4:]]
            
            # Check that the size of the signals meet the threshold
            dimension_signal = all([len(signal) == frames_video for signal in row])
            
            # Check whether zero speeds in the list meets the threshold
            zero_exceeds_threshold = any((np.sum(np.array(speed) == 0.0)) / len(speed) > zero_threshold for speed in speeds)
            
            # If the signal is invalid, mark the entire video as invalid and store the reason
            if zero_exceeds_threshold or not dimension_signal:
                invalid_videos.add(tuple(samples[i]))  # Tuple of (patient, video, clip number, resolution)

                # Append invalid video info with reason
                reason = ""
                if not dimension_signal:
                    reason = f"Dimensioni del segnale non valide (attese {frames_video} frame)"
                elif zero_exceeds_threshold:
                    reason = f"Velocità zero in più del {zero_threshold*100}% dei frame"

                invalid_video_info.append({
                    'video': samples[i],  # Patient, video, clip information
                    'reason': reason
                })

        # Remove invalid videos
        for i, sample in enumerate(samples):
            # Check if 'video' information is stored at the correct index
            if tuple(sample[::]) in invalid_videos:  # Adjust the slicing based on your actual data structure
                valid_indices.discard(i)

        # Convert valid_indices to a sorted list
        valid_indices = sorted(list(valid_indices))

        # Filter the dictionary based on valid indices
        filtered_data = {}
        for key, value in self.extr_data.items():
            if key == "signals":
                filtered_data[key] = [value[i] for i in valid_indices]
            else:
                filtered_data[key] = value[valid_indices]

        # Signals list of lists to Multidimensional numpy array
        try:
            filtered_data['signals'] = np.array(filtered_data['signals'])
        except Exception as e:
            print(f"Error while converting signals to numpy array: {e}")
        
        return filtered_data, invalid_video_info
    
    # Funzione per la normalizzazione dei segnali
    def normalization_signals(self):
        '''
        Normalizza i segnali traslando la media a zero per ciascuna feature (es: X, Y, velocità)
        all'interno di ciascun campione.

        Returns:
        - list of lists: I segnali normalizzati, con la media traslata a zero per ogni feature.
        '''

        # self.extr_data is a (n, 8, 150) numpy array
        # mean/std normlize each signal
        # Subtract the mean from each signal
        signals = self.extr_data["signals"]
        eyes_mean = np.mean(signals[:, :4, :], axis=2, keepdims=True)
        left_mean = eyes_mean[:, :2, :]
        right_mean = eyes_mean[:, 2:, :]
        distances = np.linalg.norm(left_mean - right_mean, axis=1, keepdims=True)


        zero_mean_signals = signals - np.mean(signals, axis=2, keepdims=True)
        normalized_signals = zero_mean_signals / distances
        
        
        normalized_signals = normalized_signals / np.array(self.std).reshape(1,-1,1)

        return torch.from_numpy(normalized_signals).float()


class NystInferenceDataset(NystDataset):
    def __init__(self, windows, std):
        self.extr_data = {'signals': windows}
        self.std = std

        self.fil_norm_data = self.normalization_signals()

    def __len__(self):
        return len(self.fil_norm_data)
    
    def __getitem__(self, idx):
        return self.fil_norm_data[idx]
    

if __name__ == '__main__':
    # Test the NystDataset class
    dataset_file = "/repo/nunziati/nyst/video_features_interp.csv"
    std = np.load("/repo/nunziati/nyst/std.npy")

    dataset = NystDataset(dataset_file, std)