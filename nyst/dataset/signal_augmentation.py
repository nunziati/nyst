import pandas as pd
import numpy as np
import json

def augment_data(data):
    """
    Augments the dataset by flipping the positions and speeds.

    Arguments:
    - data (dict): The original dataset dictionary containing signals, samples, and labels.
    
    Returns:
    - dict: The updated dataset dictionary with augmented data.
    """
    signals = data['signals']
    samples = data['samples']
    labels = data['labels']
    resolutions = data['resolutions']
    patients = data['patients']
    

    augmented_signals = []
    augmented_samples = []
    augmented_labels = []
    augmented_resolutions = []
    augmented_patients = []

    for i, signal in enumerate(signals):
        # Flip positions and speeds
        flipped_signal = [-np.array(signal[0]), -np.array(signal[1]),  # left_position
            -np.array(signal[2]), -np.array(signal[3]),  # right_position
            -np.array(signal[4]), -np.array(signal[5]),  # left_speed
            -np.array(signal[6]), -np.array(signal[7])   # right_speed
        ]
        augmented_signals.append(flipped_signal)

        # Modify the sample name
        original_sample = samples[i]
        augmented_sample = [
            original_sample[0],  # Patient number
            original_sample[1],  # Video number
            f"{original_sample[2]}-r",  # Clip number with '-r'
            original_sample[3]  # Resolution remains unchanged
        ]
        augmented_samples.append(augmented_sample)

        # Labels, patients an resolutions must be the same (or adjusted if necessary)
        augmented_labels.append(labels[i])
        augmented_resolutions.append(resolutions[i])
        augmented_patients.append(patients[i])

    # Directly add augmented data to the existing data dictionary
    data['signals'] = np.concatenate((signals, augmented_signals))
    data['samples'] = np.concatenate((samples, augmented_samples))
    data['labels'] = np.concatenate((labels, augmented_labels))
    data['resolutions'] = np.concatenate((resolutions, augmented_resolutions))
    data['patients'] = np.concatenate((patients, augmented_patients))

    return data