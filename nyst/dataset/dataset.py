import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import json
import sys




class CustomDataset(Dataset):
    def __init__(self, csv_input_file, csv_label_file, preprocess=None, augmentation=None, save_merged_csv=True, augment=True, new_csv_file='D:/nyst_labelled_videos'):
        
        print('Loading a Custom Dataset...')
        
        # Load the CSV file
        self.input_data = pd.read_csv(csv_input_file)
        self.label_data = pd.read_csv(csv_label_file)
        
        # Replace backslash with slash in both dataframes
        self.input_data['video'] = self.input_data['video'].str.replace('\\', '/')
        self.label_data['video'] = self.label_data['video'].str.replace('\\', '/')

        # Perform the join on the 'video' column
        self.merged_data = pd.merge(self.input_data, self.label_data, on='video', how='left')
        #print(self.merged_data.head(-1))
        #print(len(self.merged_data))

        # Applies the preprocessing function if provided
        if preprocess:
            self.data = preprocess(self.merged_data)
        else:
            self.data = self.merged_data
        print('\t ---> Preprocessing step COMPLETED\n')


        # Save the merged CSV if the flag is enabled
        if save_merged_csv:
            prep_path = os.path.join(new_csv_file, 'merged_data.csv')
            self.data.to_csv(prep_path, index=False)
            print(f"Merged data saved to {prep_path}")
        
        # Exctract data into a dictionary
        self.data = self.exctraction_values(self.data)

        # Filter the invalid data             
        self.data, self.invalid_video_info = self.filtering_invalid_data(self.data)
        print('\t ---> Filtering invalid data step COMPLETED\n')

        # Data augmentation
        if augment:
            self.data = augmentation(self.data)
        print('\t ---> Augmentation step COMPLETED\n')

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
    def     exctraction_values(self, merged_data):
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
        # Extract signaks
        signals_str = merged_data[['left_position X', 'left_position Y', 
                            'right_position X', 'right_position Y',
                            'left_speed X', 'left_speed Y', 
                            'right_speed X', 'right_speed Y']].values
        
        # Convert strings into lists of float
        signals = [[self.parse_float_list(signal) for signal in row] for row in signals_str]

        # Resolutions extraction
        resolutions = merged_data['resolution'].to_numpy().reshape(-1, 1)

        # Patient information extraction
        patients = merged_data['video'].apply(lambda x: x.split('\\')[-1].split('_')[0]) .to_numpy().reshape(-1, 1)

        # Sample and video information extraction
        samples = merged_data.apply(
            lambda x: [
                x['video'].split('\\')[-1].split('_')[0], # Patient number
                x['video'].split('_')[1],  # Video number
                x['video'].split('_')[2].split('.')[0],  # Clip number
                x['resolution']  # Resolution
            ], 
            axis=1
        ).to_numpy()

        # Label extraction
        labels = merged_data['label'].to_numpy().reshape(-1, 1)

        return {
            'signals': signals,
            'resolutions': resolutions,
            'patients': patients,
            'samples': samples,
            'labels': labels
        }

    # Funzione aggiornata per il filtraggio dei dati
    def filtering_invalid_data(self, dictionary_input:dict, frames_video:int = 300, zero_threshold:float = 0.2):
        '''
        Filters out invalid data/videos based on signal dimensions and zero-speed thresholds, and also removes 
        entries associated with the same patient, video, and clip number.

        Arguments:
        - dictionary_input (dict): A dictionary containing the input data.
        - frames_video (int): The expected number of frames in each signal. Defaults to 300.
        - zero_threshold (float): The threshold for filtering out signals with excessive zero speeds. Defaults to 0.2 (20%).

        Returns:
        - tuple:
            - dict: A dictionary with filtered data, maintaining the original structure but with invalid entries removed.
            - list: A list containing information about the invalid clips that were filtered out, along with the reasons for the filtering.
        '''

        # Retrieve the input signal values
        signals = dictionary_input['signals']
        samples = dictionary_input['samples']
        valid_indices = set(range(len(signals)))  # Start with all indices being valid
        invalid_video_info = []  # To store video information and reasons for filtering
        invalid_videos = set()

        # Cycle through all signals
        for i in range(len(signals)):
            
            # Extract the specific signal values
            row = signals[i]

            # Split the signals into positions and speeds
            positions = [self.parse_float_list(pos) if isinstance(pos, str) else pos for pos in row[:4]]
            speeds = [self.parse_float_list(speed) if isinstance(speed, str) else speed for speed in row[4:]]
            
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
        for key, value in dictionary_input.items():
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

    # String to list function
    def parse_float_list(self, string_value):
        """
        Converts a string representation of a list into a Python list of floats, handling 'nan' values.

        Arguments:
        - string_value (str): A string that represents a list of numerical values, which may include 'nan' as a placeholder for missing values.

        Returns:
        - float_list (list): A list of floats where 'nan' strings in the input are replaced with Python's float('nan') to represent missing values.
        """
        if isinstance(string_value, str):  # Check if it's a string
            string_value = string_value.replace('nan', 'null')
            # Use json.loads to convert a string to Python list/dictionary
            float_list = json.loads(string_value)
            # Replace 'null' with float 'nan'
            return [float('nan') if x is None else x for x in float_list]
        elif isinstance(string_value, np.ndarray):         
            # Convert the NumPy array to a list and then to a string
            string_value = str(string_value.tolist()).replace('nan', 'null')  # Replace 'nan' with 'null'
            # Use json.loads to convert the string to a Python list
            float_list = json.loads(string_value)
            return [float('nan') if x is None else x for x in float_list]
        else:
            # If it's not a string, you might want to handle it differently
            print(f"Warning: Expected string but got {type(string_value)}. Returning empty list.")
            return []
        
        

   