import numpy as np
from dataset import NystDataset

def compute_std(data):
    '''
    Compute the standard deviation of the signals in the input data.
    
    Arguments:
    - data (numpy.ndarray): The input data containing the signals.
    
    Returns:
    - numpy.ndarray: The standard deviation of the signals.
    '''
    
    # Compute the standard deviation of the signals
    std = np.std(data, axis=1)
    
    return std

def load_data(filename):
    dataset = NystDataset(filename, 1)
    data = dataset.fil_norm_data.numpy()
    return data

def main():
    filename = 'data.csv'
    data = load_data(filename)
    std = compute_std(data)

    np.save('std.npy', std)
    
    print(f'Standard deviation of the signals: {std}')

if __name__ == '__main__':
    main()