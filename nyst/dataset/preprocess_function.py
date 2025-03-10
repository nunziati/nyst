import numpy as np 
import json
import sys
import os
from scipy.interpolate import  splrep, splev, CubicSpline

# Add the 'code' directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nyst.dataset.signal_augmentation import *

# Preprocess function using spline interpolation
def preprocess_interpolation(data, frames=150, order=2):
    '''
    Preprocess function that checks if each signal has the expected number of frames.
    If not, it applies spline interpolation (of the given order) to upsample or downsample
    the signals to the target number of frames.

    Arguments:
    - data (pandas.DataFrame): The input DataFrame containing the signal data.
    - frames (int): The target number of frames for each signal.
    - order (int): The order of the spline interpolation.

    Returns:
    - pandas.DataFrame: The preprocessed DataFrame with the interpolated signals.
    '''
    # Nested function to interpolate a single signal
    def interpolate_signal(signal, target_frames=frames, interpolation_order=order):
        # Create an array of evenly spaced values between 0 and 1 for the original signal
        x_original = np.linspace(0, 1, len(signal))

        # Create an array of evenly spaced values between 0 and 1 for the target number of frames
        x_new = np.linspace(0, 1, target_frames)

        # Create a spline representation of the original signal using the specified interpolation order
        tck = splrep(x_original, signal, k=interpolation_order)

        # Evaluate the spline at the new x-values to get the interpolated signal
        return splev(x_new, tck)

    # Iterate through each signal column in the DataFrame
    for column in ['left_position X', 'left_position Y', 'right_position X', 'right_position Y',
                'left_speed X', 'left_speed Y', 'right_speed X', 'right_speed Y']:
        
        # Apply the interpolation function to each signal in the column
        data[column] = data[column].apply(lambda x: interpolate_signal(json.loads(x)) if len(json.loads(x)) != frames else x)
    
    return data   # Return the DataFrame with the interpolated signals

# Cubic Interpolation for Signal Data
def cubic_interpolation(data, frames=300):
    '''
    This function checks if each signal has the expected number of frames. 
    If not, it applies cubic spline interpolation with continuous derivatives and controlled boundary conditions 
    to either upsample or downsample the signals to the specified number of frames.

    Arguments:
    - data (pandas.DataFrame): The input DataFrame containing the signal data.
    - frames (int): The target number of frames for each signal. Default is 300.

    Returns:
    - pandas.DataFrame: The preprocessed DataFrame with the interpolated signals.
    '''
    
    # Nested function to interpolate a single signal using cubic spline interpolation
    def interpolate_signal(signal, target_frames=frames):
        # Create an array of evenly spaced values between 0 and 1 for the original signal
        x_original = np.linspace(0, 1, len(signal))  

        # Create an array of evenly spaced values between 0 and 1 for the target number of frames
        x_new = np.linspace(0, 1, target_frames)   

        # # Generate a cubic spline with natural boundary conditions (smooth at boundaries)  
        cubic_spline = CubicSpline(x_original, signal, bc_type='natural')  
        
        return cubic_spline(x_new) # Evaluate the cubic spline at the new x-values to get the interpolated signal

    # Iterate through each signal column in the DataFrame
    for column in ['left_position X', 'left_position Y', 'right_position X', 'right_position Y',
                   'left_speed X', 'left_speed Y', 'right_speed X', 'right_speed Y']:
        
        # Apply the interpolation function to each signal in the column
        data[column] = data[column].apply(lambda x: interpolate_signal(json.loads(x)) if len(json.loads(x)) != frames else x)
    
    return data  # Return the DataFrame with the interpolated signals

