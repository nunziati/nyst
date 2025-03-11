import numpy as np
import pandas as pd

# Function to filter data based on patients
def filter_data_by_patients(data, selected_patients):
    '''
    This function creates a boolean mask to identify which data samples belong to the selected patients and
    returns a new DataFrame containing only the data from those patients.

    Arguments:
    - data (DataFrame): A DataFrame containing the dataset, including a 'video' column with patient IDs.
    - selected_patients (array-like): A list or array of patient IDs that should be used to filter the data.

    Returns:
    - filtered_data (DataFrame): A DataFrame containing only the data samples that correspond to the selected patients.
    '''
    # Create a boolean mask indicating which patients are in the specific set
    mask = data['patient_id'].isin(selected_patients)
    # Data filtered by patients
    filtered_data = data[mask]
    return filtered_data

# Print class balance for train and test sets
def print_class_balance(data, set_name):
    class_counts = data['label'].value_counts()
    print(f"Class balance for {set_name} set:")
    for label, count in class_counts.items():
        print(f"Class {label}: {count} samples")

# Shuffle the clips within each set to randomize order
def shuffle_data(data):
    # Shuffle the order of the clips
    data = data.sample(frac=1).reset_index(drop=True)
    return data

# Function to split the data into training and test sets
def split_data(file_path, perc_test=0.1):
    '''
    Splits the input dataset into training and test sets while ensuring that each patient’s data is entirely in one set.

    The function separates the dataset by patients, ensuring no common data occurs between training and test sets, 
    shuffles the patients, and splits the data based on the given test set percentage. Then, it shuffles the data 
    clips within the training and test sets for randomness.

    Arguments:
    - file_path (str): The path to the CSV file containing the dataset.
    - perc_test (float): The percentage of the total data to include in the test set (default is 0.1, or 10%).

    Returns:
    - None: The function saves the training and test sets to 'train_label.csv' and 'test_label.csv' respectively.
    '''
    # Load the data from the CSV file
    data = pd.read_csv(file_path)
    
    # Extract unique patients
    data['patient_id'] = data['video'].str.split('_').str[0][-3:]
    unique_patients = data['patient_id'].unique()

    # Shuffle the patients to ensure randomness
    np.random.shuffle(unique_patients)
    
    # Initialize test and training sets
    test_patients = []
    current_test_size = 0
    total_size = len(data)
    
    # Distribute patients into test set until the percentage is approximately met
    for patient in unique_patients:
        if current_test_size / total_size < perc_test:
            test_patients.append(patient)
            # Update the current test size
            patient_mask = data['patient_id'] == patient
            current_test_size += np.sum(patient_mask)
        else:
            break

    # The remaining patients are for training
    train_patients = [patient for patient in unique_patients if patient not in test_patients]
    
    # Filter the data into train and test sets based on the selected patients
    test_data = filter_data_by_patients(data, test_patients)
    train_data = filter_data_by_patients(data, train_patients)
    
    # Shuffled data
    train_data = shuffle_data(train_data)
    test_data = shuffle_data(test_data)
    
    # Save the train and test sets to CSV files
    train_data.to_csv('train_label.csv', index=False)
    test_data.to_csv('test_label.csv', index=False)
    
    print_class_balance(train_data, "train")
    print_class_balance(test_data, "test")
# Example usage
# split_data('path_to_your_file.csv')

if __name__ == "__main__":
    split_data('.csv')