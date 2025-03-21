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
    std = np.std(data, axis=(0,2))
    
    
    return std

def load_data(filename, filename_std):
   # std = np.load(filename_std)
    dataset = NystDataset(filename, 1)
    data = dataset.fil_norm_data.numpy()
    return data

def main():
    filename = '/repo/porri/nyst_labelled_videos/train_dataset.csv'
    filename_std = '/repo/porri/nyst_labelled_videos/std.npy'
    data = load_data(filename, filename_std)
    std = compute_std(data)

    np.save('/repo/porri/nyst_labelled_videos/std.npy', std)
    
    print(f'Standard deviation of the signals: {std}')

if __name__ == '__main__':
    main()