import numpy as np

class FirstSpeedExtractor:
    def __init__(self, time_resolutions=[3]):
        # These represent the number of frames over which the speed will be calculate
        self.time_resolutions = time_resolutions # Frame window resolution

    def apply(self, positions:np.array, fps:int=None) -> np.array:
        '''
        Calculate speed for the given positions at various time resolutions.
        
        Arguments:
        - positions (np.array): An array of (x, y) coordinates representing positions over frames.
        - fps (int): Frames per second of the video. If provided, the speed will be adjusted to position/second.

        Returns:
        - speed_dict (dict): A dictionary where each key is a time resolution and the value is the speed array for that resolution.
        '''
        # Create an empty dictionary to store speed calculations
        speed_dict = {}

        # Iterate over each time resolution
        for time_resolution in self.time_resolutions:
            # Compute speed for the given position and time resolution
            speed_dict[time_resolution] = self.compute_speed(positions, time_resolution)

            # If fps is provided, adjust the speed by multiplying with fps
            if fps is not None:
                speed_dict[time_resolution] *= fps # Units of measurement: position/seconds 

        return speed_dict
    
    def compute_speed(self, positions:np.array, time_resolution:list) -> np.array:
        '''
        Compute the speed of positions over a specified time resolution.
        
        Arguments:
        - positions (np.array): An array of (x, y) coordinates representing positions over time.
        - time_resolution (int): The number of frames over which to calculate the speed.

        Returns:
        - speed (np.array): A 2D array where each row represents the (x, y) speed at each frame.
        '''
        # Calculate augmentation factor as half of the time resolution
        aug_factor = time_resolution // 2

        # Create arrays for augmenting the start and end of the positions array
        head = np.array([positions[0] for _ in range(aug_factor)])
        tail = np.array([positions[-1] for _ in range(aug_factor)])

        # Augment the position array by adding the head and tail to ensure the calculation window is valid for all frames.
        position_aug = np.vstack((head, positions, tail))

        # Initialize speed array with zeros
        speed = np.zeros((len(positions), 2), dtype=np.float32)  

        # Compute the speed array by iterating over each position
        for i in range(len(positions)):
            
            pos_start = position_aug[i] # Start position for the current time window
            pos_end = position_aug[(i + time_resolution) - 1] # End position for the current time window
            
            speed[i, :] = (pos_end - pos_start) / time_resolution
        
        return speed
