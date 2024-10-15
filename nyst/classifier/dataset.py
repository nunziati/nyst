import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import json


class CustomDataset(Dataset):
    def __init__(self, csv_input_file, csv_label_file, preprocess=None, transform=None):
        
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

        # Exctract data into a dictionary
        self.data = self.exctraction_values(self.data)

        # Filter the invalid data             
        self.data, self.invalid_video_info = self.filtering_invalid_data(self.data)
        print('\t ---> Filtering invalid data step COMPLETED\n')

        #print(len(self.data['signals']))
        #print(len(self.invalid_video_info)) # Multiply by 4 to obtain the correct number of row

        # Split the dataset        
        #self.train_data, self.test_data = self.split_data(self.data, 0)
        #print('\t ---> Splitting step COMPLETED\n')

        # Extract the different components of the two sets
        #self.train_signals = self.train_data['signals']
        #self.train_resolutions = self.train_data['resolutions']
        #self.train_patients = self.train_data['patients']
        #self.train_samples = self.train_data['samples']
        #self.train_labels = self.train_data['labels']
        #print(len(self.train_signals))
        #print(np.unique(self.train_patients))
        
        #self.test_signals = self.test_data['signals']
        #self.test_resolutions = self.test_data['resolutions']
        #self.test_patients = self.test_data['patients']
        #self.test_samples = self.test_data['samples']
        #self.test_labels = self.test_data['labels']
        #print(len(self.test_signals))
        #print(np.unique(self.test_patients))

        # Store the transformation function (if any)
        self.transform = transform

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
    def exctraction_values(self, merged_data):
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

    # Data filtering
    def filtering_invalid_data_prev(self, dictionary_input:dict, frames_video:int = 300, zero_threshold:float = 0.2):
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
            - set: A set containing information about the invalid clip that were filtered out.
        '''

        # Retrieve the input signal values
        signals = dictionary_input['signals']
        samples = dictionary_input['samples']
        valid_indices = set(range(len(signals)))  # Start with all indices being valid
        invalid_video_info = []
        invalid_videos = set()

        # Cycle through all signals
        for i in range(len(signals)):
            
            # Extract the specific signal values
            row = signals[i]

            # Split the signals into positions and speeds, and convert the string values into lists of float values
            positions = [self.parse_float_list(pos) if isinstance(pos, str) else pos for pos in row[:4]]
            speeds = [self.parse_float_list(speed) if isinstance(speed, str) else speed for speed in row[4:]]
                        
            # Check that the size of the signals meet the threshold
            dimension_signal = all([len(signal) == frames_video for signal in row])
                
            # Check whether zero speeds in the list meets the threshold
            zero_exceeds_threshold = any((np.sum(np.array(speed) == 0.0)) / len(speed) > zero_threshold for speed in speeds)
            
            # If the signal is invalid, mark the entire video as invalid
            if zero_exceeds_threshold or not dimension_signal:
                invalid_videos.add(tuple(samples[i][:3]))  # Tuple of (patient, video, clip number)
                invalid_video_info.append(samples[i])
        
        # Remove invalid videos
        for i, sample in enumerate(samples):
            if tuple(sample[:3]) in invalid_videos:
                valid_indices.discard(i)

        # Convert valid_indices to a sorted list
        valid_indices = sorted(list(valid_indices))

        # Filter the dictionary based on valid indices
        filtered_data = {}
        for key, value in dictionary_input.items():
            if key == "signals":
                # Filter the 'signals' key, which is a list of lists of lists with a critical dimension (dim 2) that cannot allow the transformation into a numpy array
                filtered_data[key] = [value[i] for i in valid_indices]
            else:
                # For the other keys, which are NumPy arrays, use NumPy indexing
                filtered_data[key] = value[valid_indices]

        # Signals list of lists to Multidimensional numpy array
        try:
            filtered_data['signals'] = np.array(filtered_data['signals'])
        except Exception as e:
            print(f"Error while converting signals to numpy array: {e}")
        
        return filtered_data, invalid_videos
    

    # String to list function
    def parse_float_list(self, string_value):
        """
        Converts a string representation of a list into a Python list of floats, handling 'nan' values.

        Arguments:
        - string_value (str): A string that represents a list of numerical values, which may include 'nan' as a placeholder for missing values.

        Returns:
        - float_list (list): A list of floats where 'nan' strings in the input are replaced with Python's float('nan') to represent missing values.
        """
        # Replace 'nan' with 'null' because json.loads doesn't recognize value nan
        string_value = string_value.replace('nan', 'null')
        # Use json.loads to convert a string to Python list/dictionary
        float_list = json.loads(string_value)
        # Replace 'null' with float 'nan' as in the original data
        float_list = [float('nan') if x is None else x for x in float_list]
        
        return float_list

    # Function to filter data based on patients
    def filter_data_by_patients(self, data, selected_patients):
        '''
        This function creates a boolean mask to identify which data samples belong to the selected patients and
        returns a new dictionary containing only the data from those patients.

        Arguments:
        - data (dict): A dictionary containing arrays, one of which maps each sample to a patient ID under the key 'patients'.
        - selected_patients (array-like): A list or array of patient IDs that should be used to filter the data.

        Returns:
        - filtered_data (dict): A dictionary containing only the data samples that correspond to the selected patients.
        '''
        # Create a boolean mask indicating which patients are in the specific set
        mask = np.isin(data['patients'], selected_patients)
        # Data filtered by patients
        filtered_data = {key: value[mask.flatten()] for key, value in data.items()}
        return filtered_data

    # Split the data in customised training and test sets
    def split_data(self, dictionary_input, perc_test=0.1):
        '''
        Splits the input dataset into training and test sets while ensuring that each patient’s data is entirely in one set.

        The function separates the dataset by patients, ensuring no common data occurs between training and test sets, 
        shuffles the patients, and splits the data based on the given test set percentage. Then, it shuffles the data 
        clips within both the training and test sets for randomness.

        Arguments:
        - dictionary_input (dict): A dictionary containing multiple arrays, where one of the keys is 'patients', 
                                which holds an array mapping each data sample to a patient ID.
        - perc_test (float): The percentage of the total data to include in the test set (default is 0.1, or 10%).

        Returns:
        - train_data (dict): A dictionary containing the training set, where all samples correspond to a subset of patients.
        - test_data (dict): A dictionary containing the test set, where all samples correspond to a different subset of patients.
        '''
        # Extract unique patients
        patients = dictionary_input['patients'].flatten()
        unique_patients = np.unique(patients)

        # Shuffle the patients to ensure randomness
        np.random.shuffle(unique_patients)
        
        # Initialize test and training sets
        test_patients = []
        current_test_size = 0
        total_size = len(patients)
        
        # Distribute patients into test set until the percentage is approximately met
        for patient in unique_patients:
            if current_test_size / total_size < perc_test:
                test_patients.append(patient)
                # Update the current test size
                patient_mask = patients == patient
                current_test_size += np.sum(patient_mask)
            else:
                break

        # The remaining patients are for training
        train_patients = [patient for patient in unique_patients if patient not in test_patients]
        
        # Filter the data into train and test sets based on the selected patients
        test_data = self.filter_data_by_patients(dictionary_input, test_patients)
        train_data = self.filter_data_by_patients(dictionary_input, train_patients)
        
        # Shuffle the clips within each set to randomize order
        def shuffle_data(data):
            # Shuffle the order of the clips
            indices = np.random.permutation(data['samples'].shape[0])
            # Shuffle data
            shuffled_data = {key: value[indices] for key, value in data.items()}
            return shuffled_data
        
        # Shuffled data
        train_data = shuffle_data(train_data)
        test_data = shuffle_data(test_data)
        
        return train_data, test_data



if __name__ == '__main__':
    # Replace with the actual paths to your CSV files
    csv_input_file = 'D:/nyst_labelled_videos/video_features.csv'
    csv_label_file = 'D:/nyst_labelled_videos/labels.csv'

    # Create an instance of the CustomDataset to trigger the print
    dataset = CustomDataset(csv_input_file, csv_label_file)


    # Print invalid videos info
    #print(f'\n\nThe list of invalid videos is: {dataset.invalid_video_info}')
    #print(f'\n\nThe num of invalid videos is: {len(dataset.invalid_video_info)}')
    #print(f'\n\nThe num of samples in the dataset is: {len(dataset.data["samples"])}')

    #print(f'\n\nThe num of valid videos is: {len(dataset.train_samples)}')
    #print(f'\n\nThe num of valid videos is: {len(dataset.test_samples)}')