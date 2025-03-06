import numpy as np

# Function to filter data based on patients
def filter_data_by_patients(data, selected_patients):
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

# Function to split the data into training, validation, and test sets
def split_data(dictionary_input, perc_test=0.1, perc_val=0.1):
    '''
    Splits the input dataset into training, validation, and test sets while ensuring that each patient’s data is entirely in one set.

    The function separates the dataset by patients, ensuring no common data occurs between training, validation, and test sets, 
    shuffles the patients, and splits the data based on the given test and validation set percentages. Then, it shuffles the data 
    clips within the training, validation, and test sets for randomness.

    Arguments:
    - dictionary_input (dict): A dictionary containing multiple arrays, where one of the keys is 'patients', 
                            which holds an array mapping each data sample to a patient ID.
    - perc_test (float): The percentage of the total data to include in the test set (default is 0.1, or 10%).
    - perc_val (float): The percentage of the total data to include in the validation set (default is 0.1, or 10%).

    Returns:
    - train_data (dict): A dictionary containing the training set, where all samples correspond to a subset of patients.
    - val_data (dict): A dictionary containing the validation set, where all samples correspond to a different subset of patients.
    - test_data (dict): A dictionary containing the test set, where all samples correspond to a different subset of patients.
    '''
    # Extract unique patients
    patients = dictionary_input['patients'].flatten()
    unique_patients = np.unique(patients)

    # Shuffle the patients to ensure randomness
    np.random.shuffle(unique_patients)
    
    # Initialize test, validation, and training sets
    test_patients = []
    val_patients = []
    current_test_size = 0
    current_val_size = 0
    total_size = len(patients)
    
    # Distribute patients into test set until the percentage is approximately met
    for patient in unique_patients:
        if current_test_size / total_size < perc_test:
            test_patients.append(patient)
            # Update the current test size
            patient_mask = patients == patient
            current_test_size += np.sum(patient_mask)
        elif current_val_size / total_size < perc_val:
            val_patients.append(patient)
            # Update the current validation size
            patient_mask = patients == patient
            current_val_size += np.sum(patient_mask)
        else:
            break

    # The remaining patients are for training
    train_patients = [patient for patient in unique_patients if patient not in test_patients and patient not in val_patients]
    
    # Filter the data into train, validation, and test sets based on the selected patients
    test_data = filter_data_by_patients(dictionary_input, test_patients)
    val_data = filter_data_by_patients(dictionary_input, val_patients)
    train_data = filter_data_by_patients(dictionary_input, train_patients)
    
    # Shuffle the clips within each set to randomize order
    def shuffle_data(data):
        # Shuffle the order of the clips
        indices = np.random.permutation(data['samples'].shape[0])
        # Shuffle data
        shuffled_data = {key: value[indices] for key, value in data.items()}
        return shuffled_data
    
    # Shuffled data
    train_data = shuffle_data(train_data)
    val_data = shuffle_data(val_data)
    test_data = shuffle_data(test_data)
    
    return train_data, val_data, test_data
