import numpy as np 
import json
from scipy.interpolate import interp1d, splrep, splev

# Funzione di preprocess che usa l'interpolazione spline
def preprocess_interpolation(data, frames=300, order=2):
    '''
    Preprocess function that checks if each signal has the expected number of frames.
    If not, it applies a spline interpolation (of the given order) to upsample or downsample
    the signals to the target number of frames.

    Arguments:
    - data (pandas.DataFrame): The input DataFrame with signal data.
    - frames (int): The target number of frames for each signal.
    - order (int): The order of the spline interpolation.

    Returns:
    - pandas.DataFrame: The preprocessed DataFrame with interpolated signals.
    '''
    def interpolate_signal(signal, target_frames=frames, interpolation_order=order):
        # Creazione di una funzione di interpolazione spline
        x_original = np.linspace(0, 1, len(signal))
        x_new = np.linspace(0, 1, target_frames)
        tck = splrep(x_original, signal, k=interpolation_order)
        return splev(x_new, tck)

    # Applica l'interpolazione alle colonne di segnale
    for column in ['left_position X', 'left_position Y', 'right_position X', 'right_position Y',
                'left_speed X', 'left_speed Y', 'right_speed X', 'right_speed Y']:
        data[column] = data[column].apply(lambda x: interpolate_signal(json.loads(x)) if len(json.loads(x)) != frames else x)
    
    return data